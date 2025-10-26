import os
import logging
from copy import deepcopy
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import model as MODEL_MOD
from model import AffinityPredictor
from data_utils import ProcessedDataset, collate_fn_for_readout
from typing import List, Tuple, Dict

# ========= 基本配置 =========
DATASET = "sabdab"  # 可选: "abbind" | "skempi2" | "sabdab"
PT_PATHS = {
    "abbind":  "./outdata/graph/abbind_graphs.pt",
    "skempi2": "./outdata/graph/skempi2_graphs.pt",
    "sabdab":  "./outdata/graph/sabdab_graphs.pt",
}
PT_PATH = PT_PATHS[DATASET]

# 预测 CSV 输出目录
PRED_DIR = "./outdata/predict/{DATASET}"
os.makedirs(PRED_DIR, exist_ok=True)

# ========= 训练超参 =========
SEED = 42
EPOCHS = 100
BATCH_SIZE = 8
VAL_RATIO = 0.1  # 从每折训练部分再切出验证比例
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# 辅助损失 / 约束（突变集时才起作用；sabdab 会被置 0）
LAMBDA_RANK = 0.10
RANK_MARGIN = 0.20
LAMBDA_GATE = 0.0
LAMBDA_DDG = 0.30
DDG_AMP_PEN = 1e-3

# ========= 数据集特定开关 =========
if DATASET == "sabdab":
    MODEL_MOD.ABLATE_DDG = True
    MODEL_MOD.FIX_GATE_05 = True
    LAMBDA_DDG = 0.0
    LAMBDA_GATE = 0.0
else:
    MODEL_MOD.ABLATE_DDG = False if hasattr(MODEL_MOD, "ABLATE_DDG") else False
    MODEL_MOD.FIX_GATE_05 = False if hasattr(MODEL_MOD, "FIX_GATE_05") else False

# ========= 日志：仅控制台 =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ========= 工具 =========
def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def move_graphs_to_device(batch, device):
    graph_keys = [
        "light_graphs", "heavy_graphs", "antigen_graphs",
        "mutant_subgraph_L", "mutant_subgraph_H", "mutant_subgraph_A",
        "wild_subgraph_L", "wild_subgraph_H", "wild_subgraph_A",
    ]
    for k in graph_keys:
        if k in batch and hasattr(batch[k], "to"):
            batch[k] = batch[k].to(device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.device != device:
            batch[k] = v.to(device)
    return batch

def _trimmed_mean(xs: List[float]) -> float:
    if len(xs) < 3:
        return float(np.mean(xs)) if xs else 0.0
    arr = np.sort(np.array(xs, dtype=float))
    return float(np.mean(arr[1:-1]))

def _trimmed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    n = len(y_true)
    if n < 3:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if n > 0 else 0.0
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 else 0.0
        return rmse, corr
    abs_err = np.abs(y_true - y_pred)
    best_idx = int(np.argmin(abs_err)); worst_idx = int(np.argmax(abs_err))
    mask = np.ones(n, dtype=bool); mask[best_idx] = False; mask[worst_idx] = False
    y_t, y_p = y_true[mask], y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p))) if len(y_t) > 0 else 0.0
    corr = float(np.corrcoef(y_t, y_p)[0, 1]) if len(y_t) > 1 else 0.0
    return rmse, corr

def compute_metrics(model, dataloader, device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss, total_gate, n = 0.0, 0.0, 0
    preds, trues, dgs, ddgs = [], [], [], []
    preds_mut, trues_mut = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_graphs_to_device(batch, device)
            dg_pred, ddg_pred, gate = model(batch, return_all=True)
            sample_type = batch["sample_type"].float().view(-1)
            delta_g_pred = dg_pred + ddg_pred * sample_type
            loss = criterion(delta_g_pred, batch["delta_g"])
            bsz = delta_g_pred.size(0)
            total_loss += loss.item() * bsz; total_gate += gate.mean().item() * bsz; n += bsz
            preds.append(delta_g_pred.detach().cpu().numpy())
            trues.append(batch["delta_g"].detach().cpu().numpy())
            dgs.append(dg_pred.detach().cpu().numpy())
            ddgs.append(ddg_pred.detach().cpu().numpy())
            mask = (sample_type > 0.5)
            if mask.any():
                preds_mut.append(delta_g_pred[mask].detach().cpu().numpy())
                trues_mut.append(batch["delta_g"][mask].detach().cpu().numpy())
    if len(preds) == 0:
        return {"Loss":0.0,"RMSE":0.0,"MAE":0.0,"Corr":0.0,"Trim_RMSE":0.0,"Trim_Corr":0.0,
                "RMSE_mut":0.0,"Corr_mut":0.0,"DG_mean":0.0,"DDG_mean":0.0,"Gate_mean":0.0}
    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
    dg_all = np.concatenate(dgs); ddg_all = np.concatenate(ddgs)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) > 0 else 0.0
    mae  = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    trim_rmse, trim_corr = _trimmed_metrics(y_true, y_pred)
    if len(preds_mut) > 0:
        y_pred_m = np.concatenate(preds_mut); y_true_m = np.concatenate(trues_mut)
        rmse_m = float(np.sqrt(mean_squared_error(y_true_m, y_pred_m))) if len(y_true_m) > 0 else 0.0
        corr_m = float(np.corrcoef(y_true_m, y_pred_m)[0, 1]) if len(y_true_m) > 1 else 0.0
    else:
        rmse_m, corr_m = 0.0, 0.0
    return {"Loss": total_loss / max(1, n), "RMSE": rmse, "MAE": mae, "Corr": corr,
            "Trim_RMSE": trim_rmse, "Trim_Corr": trim_corr,
            "RMSE_mut": rmse_m, "Corr_mut": corr_m,
            "DG_mean": float(dg_all.mean()) if dg_all.size else 0.0,
            "DDG_mean": float(ddg_all.mean()) if ddg_all.size else 0.0,
            "Gate_mean": total_gate / max(1, n)}

def evaluate(model, dataloader, device):
    return compute_metrics(model, dataloader, device, nn.MSELoss())

# ========= 单折训练（val 选优，test 只评一次并导出 CSV） =========
def train_one_fold(fold_id: int, train_loader, val_loader, test_loader, device):
    logging.info(f"[Fold {fold_id:02d}] Start training")
    model = AffinityPredictor().to(device)

    # 给 gate 分组不同 lr（可选）
    backbone_params, gate_params = [], []
    for n, p in model.named_parameters():
        (gate_params if ("gate_mlp" in n or "gate_temp" in n) else backbone_params).append(p)

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": gate_params, "lr": LEARNING_RATE * 0.5, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None  # 只保存在内存

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = move_graphs_to_device(batch, device)  # 确保数据在正确的设备上
            dg_pred, ddg_pred, gate = model(batch, return_all=True)  # 获取预测值
            delta_g = batch["delta_g"].view(-1)
            sample_type = batch["sample_type"].float().view(-1)

            # 计算损失
            delta_g_pred = dg_pred + ddg_pred * sample_type
            loss = criterion(delta_g_pred, delta_g)

            total_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        logging.info(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")  # 打印每个epoch的训练损失

        # 验证阶段：打印验证集的 RMSE, Corr 等
        model.eval()
        val_m = evaluate(model, val_loader, device)
        logging.info(f"[Fold {fold_id:02d} | Epoch {epoch:03d}] "
                     f"Val RMSE={val_m['RMSE']:.4f} Corr={val_m['Corr']:.4f} | "
                     f"Trim Val RMSE={val_m['Trim_RMSE']:.4f} Corr={val_m['Trim_Corr']:.4f} | "
                     f"Gate(mean)≈{val_m['Gate_mean']:.3f}")

        scheduler.step(val_m["RMSE"])

        # 仅在内存里更新 best-val
        if val_m["RMSE"] < best_val_rmse:
            best_val_rmse = val_m["RMSE"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    logging.info(f"[Fold {fold_id:02d}] Training finished. Best epoch={best_epoch}, best Val RMSE={best_val_rmse:.4f}")

    # 使用最佳验证权重评估测试集
    assert best_state is not None, "best_state is None — 训练阶段未记录到最佳验证权重"
    model.load_state_dict(best_state)

    # 测试集评估
    test_m = evaluate(model, test_loader, device)
    logging.info(f"[Fold {fold_id:02d} | FINAL TEST] RMSE={test_m['RMSE']:.4f} | PearsonR={test_m['Corr']:.4f} | "
                 f"Test(mut) RMSE={test_m['RMSE_mut']:.4f} | Test(mut) Corr={test_m['Corr_mut']:.4f}")

    # 收集预测与真实值 → CSV，附加本折 RMSE 与 PearsonR
    y_true_all, y_pred_all, stype_all = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = move_graphs_to_device(batch, device)
            dg_pred, ddg_pred, _ = model(batch, return_all=True)
            sample_type = batch["sample_type"].float().view(-1)
            delta_g_pred = dg_pred + ddg_pred * sample_type
            y_true_all.append(batch["delta_g"].detach().cpu().numpy())
            y_pred_all.append(delta_g_pred.detach().cpu().numpy())
            stype_all.append(sample_type.detach().cpu().numpy())

    y_true_all = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_all = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    stype_all = np.concatenate(stype_all) if stype_all else np.array([])

    # 保存预测结果
    csv_path = os.path.join(PRED_DIR, f"fold_{fold_id:02d}_test_predictions.csv")
    df = pd.DataFrame({
        "fold_id": fold_id,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "sample_type": stype_all,  # 0 自然样本，1 突变样本
        "fold_rmse": test_m["RMSE"],  # 本折测试集 RMSE（整列同值）
        "fold_pearsonr": test_m["Corr"]  # 本折测试集 PearsonR（整列同值）
    })
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Fold {fold_id:02d}] Saved test predictions -> {csv_path}")

    return {"fold": fold_id, "test": test_m}

# 创建数据加载器
def _build_loader(dataset, indices, device, shuffle):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda s: collate_fn_for_readout(s, device),
        drop_last=False
    )

# 主函数
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    dataset = ProcessedDataset(PT_PATH)  # 加载完整数据集
    n = len(dataset)
    logging.info(f"Total samples: {n}")

    # 十折划分
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_results = []

    indices_all = np.arange(n)

    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(indices_all), start=1):
        rng = np.random.RandomState(SEED + fold_id)
        rng.shuffle(trainval_idx)
        val_count = max(1, int(len(trainval_idx) * VAL_RATIO))
        val_idx = trainval_idx[:val_count]
        train_idx = trainval_idx[val_count:]

        train_loader = _build_loader(dataset, train_idx, device, shuffle=True)
        val_loader = _build_loader(dataset, val_idx, device, shuffle=False)
        test_loader = _build_loader(dataset, test_idx, device, shuffle=False)

        res = train_one_fold(fold_id, train_loader, val_loader, test_loader, device)
        fold_results.append(res)

    # 聚合各折测试指标
    test_rmse_list = [r["test"]["RMSE"] for r in fold_results]
    test_corr_list = [r["test"]["Corr"] for r in fold_results]
    test_rmse_mut_list = [r["test"].get("RMSE_mut", 0.0) for r in fold_results]
    test_corr_mut_list = [r["test"].get("Corr_mut", 0.0) for r in fold_results]

    logging.info(
        "[KFold Summary] "
        f"Test RMSE(mean)={np.mean(test_rmse_list):.4f} | "
        f"Test RMSE(trim-mean)={_trimmed_mean(test_rmse_list):.4f} | "
        f"Test Corr(mean)={np.mean(test_corr_list):.4f} | "
        f"Test Corr(trim-mean)={_trimmed_mean(test_corr_list):.4f} | "
        f"Test(mut) RMSE(mean)={np.mean(test_rmse_mut_list):.4f} | "
        f"Test(mut) Corr(mean)={np.mean(test_corr_mut_list):.4f}"
    )

if __name__ == "__main__":
    main()
