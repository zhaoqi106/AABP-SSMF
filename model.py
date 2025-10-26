import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

# =========================
# 消融开关（只需改这里）
# =========================
# True → 关闭 ΔΔG 头（pred_ddg 恒为 0）
ABLATE_DDG = False
# True → gate 恒为 0.5（不训练 gate）
FIX_GATE_05 = False
# True → ΔΔG 输入中的“序列差分”置零
ABLATE_SEQ_DELTA = False
# True → 互作偏置 inter_bias 置 0
ABLATE_INTER = False
# True → 忽略边特征（edge_attr 不参与）
NO_EDGE_FEATURE = False
# True → 开启“突变局部子图差分”（需要 collate 提供对应子图）
USE_MUT_REGION_ATTENTION = False


class AttnPool(nn.Module):
    """序列引导加性注意力池化：用序列向量作为上下文，对节点打分后加权聚合"""
    def __init__(self, node_dim, ctx_dim):
        super().__init__()
        self.w_q = nn.Linear(node_dim, node_dim, bias=False)
        self.w_k = nn.Linear(ctx_dim, node_dim, bias=False)
        self.v   = nn.Linear(node_dim, 1, bias=False)

    def forward(self, node_x, batch_idx, ctx_vec):  # node_x:[N,D], ctx_vec:[B,ctx]
        # 统一使用参考张量的 dtype/device，避免 AMP 下 dtype 不一致
        ctx_proj = self.w_k(ctx_vec)                       # [B,D]
        ctx_exp  = ctx_proj[batch_idx]                     # [N,D]，dtype 跟随 ctx_proj/node_x
        scores   = self.v(torch.tanh(self.w_q(node_x) + ctx_exp)).squeeze(-1)  # [N]

        max_b = int(batch_idx.max().item()) if batch_idx.numel() else -1
        attn  = scores.new_zeros(scores.shape)             # 跟随 scores 的 dtype/device
        for b in range(max_b + 1):
            m = (batch_idx == b)
            if m.any():
                # softmax 输出对齐到 attn 的 dtype，避免 index_put 报错
                attn[m] = torch.softmax(scores[m], dim=0).to(attn.dtype)

        out = node_x.new_zeros(ctx_vec.size(0), node_x.size(-1))  # [B,D]
        out.index_add_(0, batch_idx, node_x * attn.unsqueeze(-1))
        return out


class AffinityPredictor(nn.Module):
    """
    输出：
      - dg_pred: ΔG 的基线预测（所有样本）
      - ddg_pred: ΔΔG（仅突变样本有效，最终会在外部按 sample_type 融合）
      - gate: 对 ΔΔG 贡献做缩放的 gate 值（Sigmoid）
    """
    def __init__(self, node_feat_dim=23, seq_emb_dim=1280, hidden_dim=128, gate_temp=0.75):
        super().__init__()
        in_dim = node_feat_dim

        # 2 维边特征编码至 32 维（用于 TransformerConv 的 edge_attr）
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.conv = TransformerConv(in_dim, 256, heads=2, concat=False, edge_dim=32)

        self.proj1    = nn.Linear(256, hidden_dim)
        self.seq_proj = nn.Linear(seq_emb_dim, hidden_dim)

        self.pool = AttnPool(node_dim=hidden_dim, ctx_dim=hidden_dim)

        # DG 头：拼 (三链图池化 + 三链序列) 共 6H
        self.fusion_head_dg = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ΔΔG 头：输入 = 三链“结构差分”(或零) + 三链序列差分 = 6H
        self.fusion_head_ddg = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # gate：Sigmoid(MLP/temp)
        self.gate_mlp  = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.gate_temp = gate_temp

        # 轻量互作偏置（Ab↔Ag）
        self.inter_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        if USE_MUT_REGION_ATTENTION:
            self.mut_attn_weights = nn.Parameter(torch.ones(3))

    # ---- Encoders ----
    def _encode_graph_nodes(self, g):
        # 使用“全零占位的 edge_attr”以兼容 TransformerConv(edge_dim=32)
        E = g.edge_index.size(1)
        e_raw = getattr(g, "edge_attr", None)
        if NO_EDGE_FEATURE:
            e = g.x.new_zeros((E, 32))  # 跟随 g.x 的 dtype/device
        else:
            if e_raw is not None:
                e = self.edge_encoder(e_raw)               # AMP 下会自动匹配
            else:
                e = g.x.new_zeros((E, 32))                # 缺省占位，继承 dtype/device

        x = F.elu(self.conv(g.x, g.edge_index, edge_attr=e))
        h = F.dropout(self.proj1(x), 0.1, self.training)
        return h

    def _pool_with_seq(self, node_h, batch_idx, seq_vec):
        return self.pool(node_h, batch_idx, seq_vec)

    def _encode_seq(self, s):
        if s is None:
            return None
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return self.seq_proj(s)

    # ---- Forward ----
    def forward(self, batch, return_all=False):
        Lg, Hg, Ag = batch['light_graphs'], batch['heavy_graphs'], batch['antigen_graphs']
        Ls, Hs, As = batch['light_seq_emb'], batch['heavy_seq_emb'], batch['antigen_seq_emb']

        # wild 序列（自然样本可能为零向量占位）
        WLs = batch.get('wild_light_seq_emb')
        WHs = batch.get('wild_heavy_seq_emb')
        WAs = batch.get('wild_antigen_seq_emb')

        device = Lg.x.device
        sample_type = torch.as_tensor(batch['sample_type'], device=device).view(-1, 1)  # [B,1]

        # 序列编码（占位用 zeros_like，自动继承 dtype）
        L_seq  = self._encode_seq(Ls)
        H_seq  = self._encode_seq(Hs)
        A_seq  = self._encode_seq(As)
        WL_seq = self._encode_seq(WLs) if WLs is not None else (L_seq.new_zeros(L_seq.shape) if L_seq is not None else None)
        WH_seq = self._encode_seq(WHs) if WHs is not None else (H_seq.new_zeros(H_seq.shape) if H_seq is not None else None)
        WA_seq = self._encode_seq(WAs) if WAs is not None else (A_seq.new_zeros(A_seq.shape) if A_seq is not None else None)

        # 节点编码 + 序列引导池化（mutant 全局图）
        L_nodes = self._encode_graph_nodes(Lg)
        H_nodes = self._encode_graph_nodes(Hg)
        A_nodes = self._encode_graph_nodes(Ag)
        L2 = self._pool_with_seq(L_nodes, Lg.batch, L_seq)
        H2 = self._pool_with_seq(H_nodes, Hg.batch, H_seq)
        A2 = self._pool_with_seq(A_nodes, Ag.batch, A_seq)

        # DG 特征：三链图池化 + 三链序列
        L_feat = torch.cat([L2, L_seq], dim=-1)
        H_feat = torch.cat([H2, H_seq], dim=-1)
        A_feat = torch.cat([A2, A_seq], dim=-1)
        global_fused = torch.cat([L_feat, H_feat, A_feat], dim=-1)  # [B,6H]

        # 互作偏置
        inter_feat = torch.cat([L2 * A2, H2 * A2, torch.abs(L2 - A2), torch.abs(H2 - A2)], dim=-1)
        inter_bias = self.inter_head(inter_feat).view(-1)
        if ABLATE_INTER:
            inter_bias = inter_bias * 0.0

        # 结构差分（可选用局部子图）
        if USE_MUT_REGION_ATTENTION and sample_type.sum() > 0:
            mL, wL = batch["mutant_subgraph_L"], batch["wild_subgraph_L"]
            mH, wH = batch["mutant_subgraph_H"], batch["wild_subgraph_H"]
            mA, wA = batch["mutant_subgraph_A"], batch["wild_subgraph_A"]

            def enc_pool(g, seq_ctx):
                E = g.edge_index.size(1)
                e_raw = getattr(g, "edge_attr", None)
                if NO_EDGE_FEATURE:
                    e = g.x.new_zeros((E, 32))
                else:
                    e = self.edge_encoder(e_raw) if e_raw is not None else g.x.new_zeros((E, 32))
                x = F.elu(self.conv(g.x, g.edge_index, edge_attr=e))
                h = F.dropout(self.proj1(x), 0.1, self.training)
                return self.pool(h, g.batch, seq_ctx)

            mL_pool, wL_pool = enc_pool(mL, L_seq), enc_pool(wL, L_seq)
            mH_pool, wH_pool = enc_pool(mH, H_seq), enc_pool(wH, H_seq)
            mA_pool, wA_pool = enc_pool(mA, A_seq), enc_pool(wA, A_seq)

            diff_L = F.normalize(mL_pool - wL_pool, p=2, dim=-1)
            diff_H = F.normalize(mH_pool - wH_pool, p=2, dim=-1)
            diff_A = F.normalize(mA_pool - wA_pool, p=2, dim=-1)

            weights = F.softmax(self.mut_attn_weights.to(diff_L.dtype), dim=0)
            diff_L, diff_H, diff_A = diff_L * weights[0], diff_H * weights[1], diff_A * weights[2]
        else:
            diff_L = L2.new_zeros(L2.shape)
            diff_H = H2.new_zeros(H2.shape)
            diff_A = A2.new_zeros(A2.shape)

        # gate
        gate_logits = self.gate_mlp(global_fused).view(-1, 1)
        gate = torch.sigmoid(gate_logits / self.gate_temp)
        if FIX_GATE_05:
            gate = gate.new_full(gate.shape, 0.5)

        # 序列差分
        seq_delta = torch.cat([L_seq - WL_seq, H_seq - WH_seq, A_seq - WA_seq], dim=-1)
        if ABLATE_SEQ_DELTA:
            seq_delta = seq_delta.new_zeros(seq_delta.shape)

        # DG / ΔΔG 头
        pred_dg  = self.fusion_head_dg(global_fused).view(-1) + inter_bias
        ddg_input = torch.cat([diff_L, diff_H, diff_A, seq_delta], dim=-1)  # [B,6H]
        pred_ddg  = self.fusion_head_ddg(ddg_input).view(-1)
        if ABLATE_DDG:
            pred_ddg = pred_ddg * 0.0

        # 注意：模型内部不强行乘 sample_type；留给外部训练代码按需要处理
        if return_all:
            return pred_dg, pred_ddg, gate.view(-1)
        else:
            return pred_dg + pred_ddg
