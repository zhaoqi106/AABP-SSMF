import os
import warnings
import logging
import argparse
from typing import List, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from torch_geometric.data import Data, Batch

# 日志配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ===============================
# 基础特征与图构建
# ===============================
class CDRFeaturizer:
    """20维独热 + 3维理化，共23维"""
    AA_PROPS = {
        'A': [0.62, -0.5, 1.0], 'R': [-2.53, 1.0, 0.0],
        'N': [-0.78, 0.0, 0.0], 'D': [-0.90, -1.0, 0.0],
        'C': [0.29, 0.0, 1.0], 'E': [-0.74, -1.0, 0.0],
        'Q': [-0.85, 0.0, 0.0], 'G': [0.48, 0.0, 0.0],
        'H': [-0.40, 0.5, 0.0], 'I': [1.38, 0.0, 0.0],
        'L': [1.06, 0.0, 0.0], 'K': [-1.50, 1.0, 0.0],
        'M': [0.64, 0.0, 0.0], 'F': [1.19, 0.0, 0.0],
        'P': [0.12, 0.0, 0.0], 'S': [-0.18, 0.0, 0.0],
        'T': [-0.05, 0.0, 0.0], 'W': [0.81, 0.0, 0.0],
        'Y': [0.26, 0.0, 0.0], 'V': [1.08, 0.0, 0.0],
        'X': [0.0, 0.0, 0.0]
    }
    AA_ABBREV = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    def __call__(self, aa: str) -> torch.Tensor:
        one_hot = torch.zeros(len(self.AA_LIST), dtype=torch.float32)
        if aa in self.AA_LIST:
            one_hot[self.AA_LIST.index(aa)] = 1.0
        phys = torch.tensor(self.AA_PROPS.get(aa, [0, 0, 0]), dtype=torch.float32)
        return torch.cat([one_hot, phys], dim=0)


def add_distance_edge_attr(data: Data) -> Data:
    """为每条边添加到最近突变位点的序列/空间距离，并归一化到[0,1]。无突变则置零两维。"""
    E = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0
    if not hasattr(data, 'mut_idx') or data.mut_idx is None or (
            isinstance(data.mut_idx, torch.Tensor) and data.mut_idx.numel() == 0) or E == 0:
        data.edge_attr = torch.zeros((E, 2), dtype=torch.float32)
        return data

    d_spatial = torch.cdist(data.pos, data.pos[data.mut_idx], p=2)  # [N,M]
    min_sp, _ = d_spatial.min(dim=1)  # [N]

    seq_idx = data.seq_idx
    mut_idx = data.mut_idx.unsqueeze(0)
    d_seq = torch.abs(seq_idx.unsqueeze(1) - mut_idx)  # [N,M]
    min_sq, _ = d_seq.min(dim=1)  # [N]

    row, col = data.edge_index
    edge_seq = (min_sq[row] + min_sq[col]) * 0.5
    edge_spatial = (min_sp[row] + min_sp[col]) * 0.5

    # 归一化
    N = data.pos.size(0) if hasattr(data, 'pos') else 1
    edge_seq = (edge_seq / max(N - 1, 1)).clamp_(0, 1)
    max_d = float(d_spatial.max().item()) + 1e-6
    edge_spatial = (edge_spatial / max_d).clamp_(0, 1)

    data.edge_attr = torch.stack([edge_seq, edge_spatial], dim=1)  # [E,2]
    return data


class AntibodyGraphBuilder:
    def __init__(self, pdb_dir: str, contact_threshold: float = 4.5):
        self.pdb_dir = pdb_dir
        self.threshold = contact_threshold
        self.featurizer = CDRFeaturizer()
        self.pdb_parser = PDBParser(QUIET=True)

    def _load_seq_and_ca_coords(self, pdb_id: str, chain_id: str):
        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        struct = self.pdb_parser.get_structure(pdb_id, pdb_path)
        model = next(struct.get_models())
        chain = model[chain_id]
        seq, coords, resid, residue_keys = [], [], [], []
        for res in chain:
            if res.id[0] != ' ' or 'CA' not in res:
                continue
            aa = CDRFeaturizer.AA_ABBREV.get(res.get_resname(), 'X')
            seq.append(aa)
            coords.append(res['CA'].get_coord().tolist())
            resid.append(res.id[1])
            residue_keys.append((chain.id.upper(), res.id[1]))
        return seq, torch.tensor(coords, dtype=torch.float32), resid, residue_keys

    def _find_contacts(self, coords: torch.Tensor):
        dist = torch.cdist(coords, coords)
        mask = (dist > 0) & (dist < self.threshold)
        src, dst = mask.nonzero(as_tuple=True)
        return (src, dst), dist[src, dst]

    def build(self, pdb_id: str, chain_label) -> Data:
        if isinstance(chain_label, str):
            seq, coords, resid, residue_keys = self._load_seq_and_ca_coords(pdb_id, chain_id=chain_label)
        else:
            seq, coords, resid, residue_keys = [], [], [], []
            for ch in chain_label:
                s, c, r, k = self._load_seq_and_ca_coords(pdb_id, ch)
                residue_keys.extend(k)
                seq.extend(s)
                coords.append(c)
                resid.extend(r)
            coords = torch.cat(coords, dim=0) if len(coords) > 0 else torch.zeros((0, 3))

        if coords.numel() == 0 or len(seq) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
            x = torch.zeros((0, 23), dtype=torch.float32)
        else:
            (src, dst), dvals = self._find_contacts(coords)
            edge_index = torch.stack([src, dst], dim=0) if src.numel() > 0 else torch.zeros((2, 0), dtype=torch.long)
            edge_attr = dvals.unsqueeze(1) if src.numel() > 0 else torch.zeros((0, 1), dtype=torch.float32)
            x = torch.stack([self.featurizer(aa) for aa in seq], dim=0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.pos = coords if coords.numel() > 0 else torch.zeros((0, 3))
        data.seq_idx = torch.arange(x.size(0), dtype=torch.long)
        data.residue_ids = resid
        data.residue_keys = residue_keys
        return data


# ===============================
# 读写与批处理辅助
# ===============================

def build_local_subgraph(graph: Data, mut_idx: torch.Tensor, radius: float = 12.0) -> Data:
    if mut_idx is None or (isinstance(mut_idx, torch.Tensor) and mut_idx.numel() == 0) or graph.x.size(0) == 0:
        g = graph.clone()
        g.mut_idx = torch.tensor([], dtype=torch.long)
        return g

    coords = graph.pos
    center = coords[mut_idx]  # [M, 3]
    dist = torch.cdist(coords, center)  # [N, M]
    min_dist = dist.min(dim=1).values  # [N]
    selected_mask = min_dist <= radius
    selected_idx = selected_mask.nonzero(as_tuple=True)[0]

    idx_map = -torch.ones(graph.num_nodes, dtype=torch.long, device=coords.device)
    idx_map[selected_idx] = torch.arange(selected_idx.size(0), device=coords.device)

    if graph.edge_index.numel() > 0:
        row, col = graph.edge_index
        edge_mask = selected_mask[row] & selected_mask[col]
        edge_index = graph.edge_index[:, edge_mask]
        edge_index = idx_map[edge_index]
        edge_attr = graph.edge_attr[edge_mask] if (
                    hasattr(graph, 'edge_attr') and graph.edge_attr is not None) else None
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = None

    sub = Data(
        x=graph.x[selected_idx],
        pos=graph.pos[selected_idx],
        edge_index=edge_index,
        edge_attr=edge_attr,
        seq_idx=graph.seq_idx[selected_idx] if hasattr(graph, 'seq_idx') else None
    )
    if hasattr(graph, 'residue_ids'):
        sub.residue_ids = [graph.residue_ids[i] for i in selected_idx.tolist()]
    if hasattr(graph, 'residue_keys'):
        sub.residue_keys = [graph.residue_keys[i] for i in selected_idx.tolist()]

    old2new = {old.item(): new_i for new_i, old in enumerate(selected_idx)}
    mapped_idx = [old2new[i] for i in mut_idx.tolist() if i in old2new]
    sub.mut_idx = torch.tensor(mapped_idx, dtype=torch.long)

    return sub


def safe_assign_mut_idx_batch(b: Batch):
    try:
        N = b.x.size(0)
        if not hasattr(b, 'mut_idx') or b.mut_idx.numel() == 0:
            b.mut_idx = torch.tensor([], dtype=torch.long)
            b.mut_idx_batch = torch.tensor([], dtype=torch.long)
            return
        valid_mask = (b.mut_idx >= 0) & (b.mut_idx < N)
        valid_mut_idx = b.mut_idx[valid_mask]
        b.mut_idx = valid_mut_idx
        b.mut_idx_batch = b.batch[valid_mut_idx] if valid_mut_idx.numel() > 0 else torch.tensor([], dtype=torch.long)
    except Exception as e:
        logging.warning(f"[safe_assign_mut_idx_batch] {e}")
        b.mut_idx = torch.tensor([], dtype=torch.long)
        b.mut_idx_batch = torch.tensor([], dtype=torch.long)


def collate_fn_for_readout(samples: List[Dict], device: torch.device) -> Dict:
    """
    兼容两种数据形态：
    - 自然数据集：仅含 wild_* 字段；本函数会自动用 wild_* 克隆补齐 mutant 三条链与序列；
    - 突变数据集：同时含 mutant 与 wild；按原逻辑使用。

    下游模型与训练脚本无需修改：sample_type=0 时 ΔΔG 与 gate 自动不参与损失。
    """
    from torch_geometric.data import Data

    # —— 1) 自然集兜底：把缺失的 mutant 键用 wild_* 填充；缺失 sample_type 默认 0 —— #
    for s in samples:
        if "sample_type" not in s:
            s["sample_type"] = 0.0  # 自然集默认 0

        # 三条链图：若缺 mutant，就克隆 wild_*；极端情况下给空图兜底
        for plain, wild in [("light", "wild_light"), ("heavy", "wild_heavy"), ("antigen", "wild_antigen")]:
            if f"{plain}_graph" not in s or s[f"{plain}_graph"] is None:
                if f"{wild}_graph" in s and isinstance(s[f"{wild}_graph"], Data):
                    s[f"{plain}_graph"] = s[f"{wild}_graph"].clone()
                else:
                    s[f"{plain}_graph"] = Data(
                        x=torch.zeros((0, 23), dtype=torch.float32),
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                        edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                        pos=torch.zeros((0, 3), dtype=torch.float32),
                    )

        # 三条链序列：若缺 mutant，就用 wild_*；再不行给 1280 维零向量
        for plain, wild in [("light", "wild_light"), ("heavy", "wild_heavy"), ("antigen", "wild_antigen")]:
            if f"{plain}_seq_emb" not in s or s[f"{plain}_seq_emb"] is None:
                if f"{wild}_seq_emb" in s and s[f"{wild}_seq_emb"] is not None:
                    s[f"{plain}_seq_emb"] = s[f"{wild}_seq_emb"]
                else:
                    s[f"{plain}_seq_emb"] = torch.zeros(1280)

    # —— 2) 规范化：补边特征 + 把 x 搬到 device（mutant 与 wild 都处理） —— #
    for s in samples:
        # mutant 三条链
        for key in ["light_graph", "heavy_graph", "antigen_graph"]:
            s[key] = add_distance_edge_attr(s[key])
            s[key].x = s[key].x.to(device)
        # wild 三条链（若存在）
        for key in ["wild_light_graph", "wild_heavy_graph", "wild_antigen_graph"]:
            if key in s and isinstance(s[key], Data):
                s[key] = add_distance_edge_attr(s[key])
                s[key].x = s[key].x.to(device)

    def _stack_or_zeros(field: str):
        if field in samples[0]:
            return torch.stack([s.get(field, torch.zeros(1280)) for s in samples]).to(device)
        else:
            return torch.zeros((len(samples), 1280), device=device)

    # —— 3) 组装 batch 张量 —— #
    batch = {"light_graphs": Batch.from_data_list([s["light_graph"] for s in samples]),
             "heavy_graphs": Batch.from_data_list([s["heavy_graph"] for s in samples]),
             "antigen_graphs": Batch.from_data_list([s["antigen_graph"] for s in samples]),
             "light_seq_emb": _stack_or_zeros("light_seq_emb"), "heavy_seq_emb": _stack_or_zeros("heavy_seq_emb"),
             "antigen_seq_emb": _stack_or_zeros("antigen_seq_emb"),
             "wild_light_seq_emb": _stack_or_zeros("wild_light_seq_emb"),
             "wild_heavy_seq_emb": _stack_or_zeros("wild_heavy_seq_emb"),
             "wild_antigen_seq_emb": _stack_or_zeros("wild_antigen_seq_emb"),
             "delta_g": torch.stack([s["delta_g"] for s in samples]).to(device),
             "sample_type": torch.tensor([s.get("sample_type", 1.0) for s in samples],
                                         dtype=torch.float32, device=device),
             "ddg": torch.stack([s.get("ddg", torch.tensor(0.0)) for s in samples]).to(device)}

    # ddg（没有就补 0；自然集通常没有）

    # —— 4) 构造 mutant / wild 的局部子图（L/H/A），半径 12 —— #
    for short, key in zip(["L", "H", "A"], ["light", "heavy", "antigen"]):
        mutant_subs, wild_subs = [], []
        for s in samples:
            g_mut = s[f"{key}_graph"]
            mut_idx = getattr(g_mut, "mut_idx", torch.tensor([], dtype=torch.long))
            mutant_subs.append(build_local_subgraph(g_mut, mut_idx, radius=12.0))

            # 优先使用真实 wild 图；若不存在退回克隆（清空 mut_idx）
            if f"wild_{key}_graph" in s and isinstance(s[f"wild_{key}_graph"], Data):
                g_w = s[f"wild_{key}_graph"]
                w_mut_idx = getattr(g_w, "mut_idx", torch.tensor([], dtype=torch.long))
                wild_subs.append(build_local_subgraph(g_w, w_mut_idx, radius=12.0))
            else:
                g_w = g_mut.clone()
                if hasattr(g_w, "x") and g_w.x.size(1) > 23:
                    g_w.x[:, 23:] = 0
                g_w.mut_idx = torch.tensor([], dtype=torch.long)
                g_w = add_distance_edge_attr(g_w)
                wild_subs.append(build_local_subgraph(g_w, g_w.mut_idx, radius=12.0))

        # 打包 Batch + 修正 mut_idx_batch
        b_mut = Batch.from_data_list(mutant_subs)
        b_wld = Batch.from_data_list(wild_subs)
        safe_assign_mut_idx_batch(b_mut)
        safe_assign_mut_idx_batch(b_wld)

        batch[f"mutant_subgraph_{short}"] = b_mut
        batch[f"wild_subgraph_{short}"]   = b_wld

    return batch



class ProcessedDataset(Dataset):
    """直接加载 .pt 列表；（已在生成时内嵌了序列向量），此处只做透传。"""

    def __init__(self, pt_path: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.data_list = torch.load(pt_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def _parse_mut_list(val) -> List[int]:
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    s = s.replace(';', ',')
    try:
        return [int(x) for x in s.split(',') if x.strip()]
    except Exception:
        return []


def _map_resids_to_idx(residue_ids: List[int], want_resids: List[int]) -> torch.Tensor:
    if not want_resids:
        return torch.tensor([], dtype=torch.long)
    idx_map = {resid: i for i, resid in enumerate(residue_ids)}
    found = [idx_map[r] for r in want_resids if r in idx_map]
    return torch.tensor(found, dtype=torch.long)


def _load_seq_dict(path: str) -> Dict[str, torch.Tensor]:
    """健壮加载：优先 weights_only=True，失败则退回普通 load。"""
    if not path:
        return {}
    if not os.path.exists(path):
        logging.warning(f"seq embedding file not found: {path}")
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)
    except Exception as e:
        logging.error(f"load seq embeddings failed: {e}")
        return {}


def _get_emb(feats: Dict[str, torch.Tensor], key: str, dim: int = 1280) -> torch.Tensor:
    emb = feats.get(key)
    if emb is None:
        return torch.zeros(dim)
    return emb.squeeze(0) if hasattr(emb, 'dim') and emb.dim() > 1 else emb


def build_graph_pair(
        mut_builder: Optional[AntibodyGraphBuilder],
        wild_builder: AntibodyGraphBuilder,
        row_mut: Optional[pd.Series],
        row_wild: pd.Series,
        seq_mut: Dict[str, torch.Tensor],
        seq_wld: Dict[str, torch.Tensor],
        use_mutant: bool
) -> Dict:
    """
    构建包含突变图和/或野生图的数据字典。
    - 突变模式 (use_mutant=True):
        * 野生型：用 CSV 中突变条目的前4位去匹配“突变前”的 PDB（无后缀）
        * 突变体：用 CSV 中的完整 pdb 名（含后缀）
    - 自然模式 (use_mutant=False):
        * 野生型：严格按 CSV 中的完整 pdb 名（含后缀）
    """
    if row_wild is None:
        logging.error("Error: row_wild is None, skipping processing")
        return {}

    pdb_full = str(row_wild["pdb"]).strip()   # 例如 1BJ1_1
    pdb4 = pdb_full[:4]                        # 例如 1BJ1

    item = {
        "pdb": pdb_full,
        "Lchain": row_wild.get("Lchain"),
        "Hchain": row_wild.get("Hchain"),
        "antigen_chain": row_wild.get("antigen_chain"),
        "delta_g": torch.tensor(float(row_wild.get("delta_g", 0.0)), dtype=torch.float32),
        "sample_type": 1.0 if use_mutant else 0.0
    }

    # ===== 野生型：按模式选择 PDB 名 =====
    # 突变模式：用前4位（“突变前”的原始 PDB 文件名）
    # 自然模式：用完整名（含后缀）
    wild_pdb_for_file = pdb4 if use_mutant else pdb_full

    for key, chain_id in (("light", row_wild["Lchain"]),
                          ("heavy", row_wild["Hchain"]),
                          ("antigen", row_wild["antigen_chain"])):
        if pd.isna(chain_id) or not chain_id:
            continue

        # 构建野生型图
        g_wld = wild_builder.build(wild_pdb_for_file, chain_id)

        # 突变位点（自然集通常为空；突变集里需要用到）
        res_col = {"light": "mut_res_L", "heavy": "mut_res_H", "antigen": "mut_res_A"}[key]
        mut_res_list = _parse_mut_list(row_wild.get(res_col))
        g_wld.mut_idx = _map_resids_to_idx(g_wld.residue_ids, mut_res_list)

        item[f"wild_{key}_graph"] = g_wld

        # 野生型序列嵌入 key：突变模式用前4位， 自然模式用完整名
        wild_key = f"{wild_pdb_for_file}_{chain_id}"
        item[f"wild_{key}_seq_emb"] = _get_emb(seq_wld, wild_key)

    # ===== 突变体：仅突变模式下构建，使用完整名（含后缀）=====
    if use_mutant and mut_builder is not None and row_mut is not None:
        for key, chain_id in (("light", row_mut["Lchain"]),
                              ("heavy", row_mut["Hchain"]),
                              ("antigen", row_mut["antigen_chain"])):
            if pd.isna(chain_id) or not chain_id:
                continue

            g_mut = mut_builder.build(pdb_full, chain_id)

            res_col = {"light": "mut_res_L", "heavy": "mut_res_H", "antigen": "mut_res_A"}[key]
            mut_res_list = _parse_mut_list(row_mut.get(res_col))
            g_mut.mut_idx = _map_resids_to_idx(g_mut.residue_ids, mut_res_list)

            item[f"{key}_graph"] = g_mut

            mut_key = f"{pdb_full}_{chain_id}"
            item[f"{key}_seq_emb"] = _get_emb(seq_mut, mut_key)

        # ΔΔG（突变 - 野生）
        wild_delta_g = float(row_wild.get("delta_g", 0.0))
        mut_delta_g = float(row_mut.get("delta_g", 0.0))
        item["ddg"] = torch.tensor(mut_delta_g - wild_delta_g, dtype=torch.float32)

    return item


def main():
    # 自动选择模式，不需要通过命令行输入
    mode = 'wild'  # 或者 'mutant'

    use_mutant = (mode == "mutant")

    # 配置路径
    config = {
        "wild_pdb": "./datasets/PDB/skempi2",
        "wild_csv": "./datasets/CSV/skempi2.csv",
        "mut_pdb": "./datasets/PDBSOURCE/skempi2",
        "mut_csv": "./datasets/CSV/skempi2_source.csv",
        "wild_seq_emb": "./outdata/seq_embeding/skempi2_embedings.pt",
        "mut_seq_emb": "./outdata/seq_embeding/skempi2_source_embedings.pt",
        "output": "./outdata/graph/{}.pt".format("mutant_graphs" if use_mutant else "skempi2_graphs")
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(config["output"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

    # 加载序列特征
    wild_seq_features = _load_seq_dict(config["wild_seq_emb"])
    mut_seq_features = _load_seq_dict(config["mut_seq_emb"]) if use_mutant else {}

    # 初始化图构建器
    wild_builder = AntibodyGraphBuilder(config["wild_pdb"])
    mut_builder = AntibodyGraphBuilder(config["mut_pdb"]) if use_mutant else None

    # 加载CSV数据
    df_wld = pd.read_csv(config["wild_csv"])
    df_mut = pd.read_csv(config["mut_csv"]) if use_mutant else None

    # 创建索引用于对齐数据
    if use_mutant:
        # 在突变模式下，需要对齐突变体和野生型
        wild_index = {row["pdb"][:4]: row for _, row in df_wld.iterrows()}
    else:
        # 在自然数据集模式下，直接处理野生型
        wild_index = None

    items = []
    missed = 0

    if use_mutant:
        # ===========================================
        # 突变数据集处理模式
        # ===========================================
        for i, row in df_mut.iterrows():
            try:
                pdb_id = str(row["pdb"]).strip()
                key4 = pdb_id[:4]

                # 查找对应的野生型数据
                if key4 not in wild_index:
                    logging.warning(f"Wild type for {pdb_id} not found, skipping")
                    missed += 1
                    continue

                row_wild = wild_index[key4]

                item = build_graph_pair(
                    mut_builder=mut_builder,
                    wild_builder=wild_builder,
                    row_mut=row,
                    row_wild=row_wild,
                    seq_mut=mut_seq_features,
                    seq_wld=wild_seq_features,
                    use_mutant=use_mutant
                )

                if item:
                    items.append(item)

                if i % 100 == 0:
                    logging.info(f"Processed {i} mutant samples")

            except Exception as e:
                logging.error(f"Error processing mutant row {i} (PDB: {pdb_id}): {str(e)}")
    else:
        # ===========================================
        # 自然数据集处理模式
        # ===========================================
        for i, row in df_wld.iterrows():
            try:
                pdb_id = str(row["pdb"]).strip()

                item = build_graph_pair(
                    mut_builder=None,
                    wild_builder=wild_builder,
                    row_mut=None,
                    row_wild=row,
                    seq_mut={},
                    seq_wld=wild_seq_features,
                    use_mutant=use_mutant
                )

                if item:
                    items.append(item)

                if i % 100 == 0:
                    logging.info(f"Processed {i} wild samples")

            except Exception as e:
                pdb_id = row["pdb"] if "pdb" in row else "unknown"
                logging.error(f"Error processing wild row {i} (PDB: {pdb_id}): {str(e)}")

    # 保存结果
    try:
        torch.save(items, config["output"])
        logging.info(f"Successfully saved {len(items)} items to {config['output']}")
        if missed > 0:
            logging.warning(f"{missed} items skipped due to missing wild type data")
    except Exception as e:
        logging.error(f"Error saving .pt file: {e}")


if __name__ == "__main__":
    main()
