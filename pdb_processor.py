import os
import logging
import torch
from typing import Dict, Optional, List
from Bio import SeqIO
import esm
import tqdm
import re
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SequenceFeatureExtractor:
    """
    从FASTA文件中提取蛋白质序列特征，使用本地ESM2模型
    """

    def __init__(self, fasta_path: str, esm_model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化序列特征提取器
        """
        self.fasta_path = fasta_path
        self.sequence_dict = self._load_fasta()
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.feature_cache = {}  # 特征缓存

        # 加载ESM2模型
        self.esm_model = None
        self.esm_alphabet = None
        if esm_model_path:
            self._load_esm_model(esm_model_path)

    def _load_esm_model(self, model_path: str):
        """加载本地ESM2模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logging.error(f"ESM模型文件不存在: {model_path}")
                return

            # 使用esm.pretrained加载模型
            self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
            self.esm_model = self.esm_model.eval().to(self.device)

            # 获取模型信息
            model_name = os.path.basename(model_path)
            num_params = sum(p.numel() for p in self.esm_model.parameters())

            logging.info(f"成功加载本地ESM2模型: {model_name}")
            logging.info(f"模型位置: {model_path}")
            logging.info(f"模型参数: {num_params:,}")
            logging.info(f"字母表大小: {len(self.esm_alphabet)}")
        except Exception as e:
            logging.error(f"加载ESM模型失败: {str(e)}")
            self.esm_model = None

    def _load_fasta(self) -> Dict[str, str]:
        """
        加载FASTA文件到字典
        """
        sequence_dict = {}
        try:
            with open(self.fasta_path, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    # 移除header中的版本号（如.1）以匹配PDB ID
                    clean_id = re.sub(r'\.\d+$', '', record.id)
                    sequence_dict[clean_id] = str(record.seq)
            logging.info(f"成功加载 {len(sequence_dict)} 条序列从 {self.fasta_path}")
        except Exception as e:
            logging.error(f"加载FASTA文件失败: {str(e)}")
            raise

        return sequence_dict

    def get_sequence(self, pdb_id: str, chain_type: str) -> str:
        """
        获取特定蛋白质序列
        """
        # 构建FASTA头标识符
        header = f"{pdb_id}_{chain_type}"

        # 尝试获取序列
        sequence = self.sequence_dict.get(header, "")

        if not sequence:
            logging.warning(f"未找到序列: {header}")

        return sequence

    def get_sequence_features(self, sequence: str) -> torch.Tensor:
        """
        从序列中提取基础特征
        """
        # 检查缓存
        if sequence in self.feature_cache:
            return self.feature_cache[sequence]

        # 氨基酸列表（20种标准氨基酸）
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        features = []

        # 1. 序列长度（标准化）
        features.append(len(sequence) / 1000)  # 除以1000进行归一化

        # 2. 氨基酸组成（百分比）
        for aa in amino_acids:
            count = sequence.count(aa)
            features.append(count / len(sequence) if len(sequence) > 0 else 0)

            # 3. 疏水性特征
            hydrophobic_aas = "AILMFWV"
            hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aas)
            features.append(hydrophobic_count / len(sequence) if len(sequence) > 0 else 0)

            # 4. 带电氨基酸比例
            charged_aas = "DEKRH"
            charged_count = sum(1 for aa in sequence if aa in charged_aas)
            features.append(charged_count / len(sequence) if len(sequence) > 0 else 0)

            # 转换为张量
            tensor_features = torch.tensor(features, dtype=torch.float32)

            # 缓存结果
            self.feature_cache[sequence] = tensor_features

        return tensor_features

    def get_esm_embedding(self, sequence: str, layer: int = 33, mean_pool: bool = True) -> torch.Tensor:
        """
        使用本地ESM2模型生成嵌入向量
        """
        # 检查缓存
        cache_key = f"{sequence}_{layer}_{mean_pool}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        if not self.esm_model:
            logging.warning("ESM模型未加载，使用基础特征")
            return self.get_sequence_features(sequence)

        try:
            # 准备输入数据
            batch_converter = self.esm_alphabet.get_batch_converter()
            batch_labels = ["seq0"]
            batch_seqs = [sequence]
            _, _, batch_tokens = batch_converter([(batch_labels[0], batch_seqs[0])])
            batch_tokens = batch_tokens.to(self.device)

            # 提取嵌入
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[layer])

            # 获取指定层的表示
            token_representations = results["representations"][layer]

            # 移除特殊标记 (CLS, SEP, PAD)
            # ESM2的特殊标记: <cls> (索引0), <eos> (索引2), <pad> (索引1)
            seq_len = len(sequence)
            token_representations = token_representations[:, 1:seq_len + 1, :]  # (1, L, D)

            # 平均池化
            if mean_pool:
                embedding = token_representations.mean(dim=1).squeeze(0).cpu()
            else:
                embedding = token_representations.squeeze(0).cpu()

            # 缓存结果
            self.feature_cache[cache_key] = embedding
            return embedding

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.warning(f"GPU内存不足，尝试使用CPU处理序列: {sequence[:10]}...")
                return self._process_on_cpu(sequence, layer, mean_pool)
            else:
                logging.error(f"ESM嵌入生成失败: {str(e)}")
                return self.get_sequence_features(sequence)
        except Exception as e:
            logging.error(f"ESM嵌入生成失败: {str(e)}")
            return self.get_sequence_features(sequence)

    def _process_on_cpu(self, sequence: str, layer: int, mean_pool: bool) -> torch.Tensor:
        """在CPU上处理序列"""
        try:
            # 临时将模型移到CPU
            original_device = next(self.esm_model.parameters()).device
            self.esm_model = self.esm_model.cpu()

            batch_converter = self.esm_alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter([("seq0", sequence)])

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[layer])

            token_representations = results["representations"][layer]
            seq_len = len(sequence)
            token_representations = token_representations[:, 1:seq_len + 1, :]

            if mean_pool:
                embedding = token_representations.mean(dim=1).squeeze(0)
            else:
                embedding = token_representations.squeeze(0)

            # 将模型移回原设备
            self.esm_model = self.esm_model.to(original_device)

            return embedding

        except Exception as e:
            logging.error(f"CPU处理失败: {str(e)}")
            return self.get_sequence_features(sequence)

    def get_batch_esm_embeddings(self, sequences: List[str], batch_size: int = 4,
                                 layer: int = 33, mean_pool: bool = True) -> List[torch.Tensor]:
        """
        批量处理序列嵌入，提高效率
        """
        if not self.esm_model:
            return [self.get_sequence_features(seq) for seq in sequences]

        embeddings = []
        batch_converter = self.esm_alphabet.get_batch_converter()

        # 分批处理
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_labels = [f"seq_{j}" for j in range(len(batch_seqs))]
            batch_data = list(zip(batch_labels, batch_seqs))

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[layer])

            token_representations = results["representations"][layer]

            for j in range(len(batch_seqs)):
                seq_len = len(batch_seqs[j])
                seq_rep = token_representations[j, 1:seq_len + 1, :]

                if mean_pool:
                    emb = seq_rep.mean(dim=0).cpu()
                else:
                    emb = seq_rep.cpu()

                embeddings.append(emb)

        return embeddings


def process_sequences(fasta_path: str, output_dir: str,
                      esm_model_path: Optional[str] = None,
                      device: str = "cuda",
                      batch_size: int = 8):
    """
    处理所有序列并保存特征
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = SequenceFeatureExtractor(
        fasta_path=fasta_path,
        esm_model_path=esm_model_path,
        device=device
    )

    # 收集所有特征
    features_dict = {}
    sequences_to_process = []
    headers_to_process = []

    # 准备处理数据
    for header in extractor.sequence_dict.keys():
        pdb_id, chain_type = header.split('_', 1)
        sequence = extractor.get_sequence(pdb_id, chain_type)
        sequences_to_process.append(sequence)
        headers_to_process.append(header)

    # 批量处理序列
    if extractor.esm_model:
        logging.info(f"使用ESM2模型批量处理 {len(sequences_to_process)} 条序列 (batch_size={batch_size})")
        embeddings = extractor.get_batch_esm_embeddings(
            sequences_to_process,
            batch_size=batch_size
        )

        for header, emb in zip(headers_to_process, embeddings):
            features_dict[header] = emb
    else:
        logging.info("使用基础序列特征")
        for header, seq in zip(headers_to_process, sequences_to_process):
            features_dict[header] = extractor.get_sequence_features(seq)

    # 保存特征
    output_path = os.path.join(output_dir, "abbind_embedings.pt")
    torch.save(features_dict, output_path)
    logging.info(f"保存序列特征到: {output_path} (共 {len(features_dict)} 条序列)")

    # 返回特征维度信息
    sample_feat = next(iter(features_dict.values()))
    logging.info(f"特征维度: {sample_feat.shape}")

    # 显式释放模型内存
    if extractor.esm_model:
        del extractor.esm_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 配置路径
    fasta_path = "../../datasets/FASTA/abbind_sequences.fasta"
    output_dir = "."
    esm_model_path = "../../esm2_t33_650M_UR50D/esm2_t33_650M_UR50D.pt"

    # 处理序列（使用ESM2高级嵌入）
    process_sequences(
        fasta_path=fasta_path,
        output_dir=output_dir,
        esm_model_path=esm_model_path,
        device="cuda",  # 如果GPU可用
        batch_size=8  # 根据GPU内存调整
    )
