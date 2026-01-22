import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
from torch import Tensor
from models.qformer import QFormer
from models.model import PeakTokenizer




class XRDClassificationModel(nn.Module):
    """
    XRD分类模型，预测晶体系统
    """
    def __init__(self,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.0,
                 ):
        super().__init__()

        self.xrd_encoder = PeakTokenizer(embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 7)  # 这里改为7
        )



        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, peaks_x: torch.Tensor, peaks_y: torch.Tensor, peaks_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            peaks_x: XRD图谱峰值的角度 [batch_size, seq_len]
            peaks_y: XRD图谱峰值的强度 [batch_size, seq_len]
            peaks_mask: XRD输入数据的掩码 [batch_size, seq_len]
        Returns:
            logits: 预测值 [batch_size, 1]
        """
        # xrd_embedding = self.xrd_encoder(xrd_input)
        xrd_seq, xrd_mask = self.xrd_encoder(peaks_x, peaks_y, peaks_mask)
        xrd_embedding = self.transformer(xrd_seq, src_key_padding_mask=xrd_mask)
        xrd_embedding = xrd_embedding[:, 0, :]

        logits = self.mlp(xrd_embedding)

        return logits

class XRDFormulaClassificationModel(nn.Module):
    """
    XRD分类模型，输入XRD图谱和化学式原子，预测晶体系统类别
    """
    def __init__(self,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.0,
                 formula_vocab_size: int = 118, # 假设有118种元素
                 num_classes: int = 7
                 ):
        super().__init__()

        self.xrd_encoder = PeakTokenizer(embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.formula_embedding = nn.Embedding(formula_vocab_size, embedding_dim) # nn.Embedding需要训练


        # [batch_size, {peak_max_len + formula_max_len}, embedding_dim]
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim + 9, 64), #  + 9 = 6(lattice) + 1(space_group) + 1(bandgap) + 1(density)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 分类任务输出类别数
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, peaks_x: torch.Tensor, peaks_y: torch.Tensor, peaks_mask: torch.Tensor, formula: torch.Tensor, formula_mask: torch.Tensor, lattice_parameter: torch.Tensor, space_group: torch.Tensor, bandgap: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            peaks_x: XRD图谱峰值的角度 [batch_size, seq_len]
            peaks_y: XRD图谱峰值的强度 [batch_size, seq_len]
            peaks_mask: XRD输入数据的掩码 [batch_size, seq_len]
            formula: 化学式 [batch_size, formula_seq_len]
            formula_mask: 化学式掩码 [batch_size, formula_seq_len]
            lattice_parameter: 晶格参数 [batch_size, 6]
            space_group: 空间群 [batch_size]
            bandgap: 能带隙 [batch_size]
            density: 密度 [batch_size]
        Returns:
            logits: 预测类别 [batch_size, num_classes]
        """
        xrd_seq, xrd_mask = self.xrd_encoder(peaks_x, peaks_y, peaks_mask) # 形状为 [batch_size, peak_max_len+1, embedding_dim]
        xrd_embedding = self.transformer(xrd_seq, src_key_padding_mask=xrd_mask) # 形状为 [batch_size, peak_max_len+1, embedding_dim]
        xrd_embedding = xrd_embedding[:, 0, :]  # 取CLS或第一个token。丢了其它token的信息。形状为 [batch_size, embedding_dim]

        formula_embedding = self.formula_embedding(formula)
        emb_dim = formula_embedding.shape[-1]
        formula_mask = torch.logical_not(formula_mask.unsqueeze(-1).repeat(1, 1, emb_dim)).to(formula_embedding.dtype)
        formula_embedding = torch.mul(formula_embedding, formula_mask)
        formula_embedding = torch.sum(formula_embedding, dim=1) # 形状为 [batch_size, emb_dim]

        # 处理这三个特征
        # 假设 space_group, bandgap 都是一维 [batch]
        # lattice_parameter: [batch, 6]
        extra_features = torch.cat([
            lattice_parameter,  # [batch, 6]
            space_group.unsqueeze(1),        # [batch, 1]。会把 shape 从 [batch] 变成 [batch, 1]，即每个样本的类别编号变成一列
            bandgap.unsqueeze(1),           # [batch, 1]
            density.unsqueeze(1)            # [batch, 1]
        ], dim=1)  # [batch, 9]

        # embedding = torch.cat([xrd_embedding, formula_embedding], dim=-1)
        # 拼接到 embedding
        embedding = torch.cat([xrd_embedding, formula_embedding, extra_features], dim=-1)

        logits = self.mlp(embedding)
        return logits



class XRDClassificationRegressionModel(nn.Module):
    """
    XRD分类模型，输入XRD图谱和化学式原子，预测晶体系统类别
    """
    def __init__(self,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.0,
                 formula_vocab_size: int = 118, # 假设有118种元素
                 num_classes: int = 7
                 ):
        super().__init__()

        self.xrd_encoder = PeakTokenizer(embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.formula_embedding = nn.Embedding(formula_vocab_size, embedding_dim) # nn.Embedding需要训练


        # [batch_size, {peak_max_len + formula_max_len}, embedding_dim]
        self.mlp1 = nn.Sequential(
            nn.Linear(2 * embedding_dim + 3, 64), #  + 3 = + 1(space_group) + 1(bandgap) + 1(density)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 分类任务输出类别数
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(2 * embedding_dim + 3, 64), #  + 3 = 1(space_group) + 1(bandgap) + 1(density)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 回归任务输出晶格参数数值 lattice_parameter 有6个参数
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, peaks_x: torch.Tensor, peaks_y: torch.Tensor, peaks_mask: torch.Tensor, formula: torch.Tensor, formula_mask: torch.Tensor, space_group: torch.Tensor, bandgap: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            peaks_x: XRD图谱峰值的角度 [batch_size, seq_len]
            peaks_y: XRD图谱峰值的强度 [batch_size, seq_len]
            peaks_mask: XRD输入数据的掩码 [batch_size, seq_len]
            formula: 化学式 [batch_size, formula_seq_len]
            formula_mask: 化学式掩码 [batch_size, formula_seq_len]
            space_group: 空间群 [batch_size]
            bandgap: 能带隙 [batch_size]
            density: 密度 [batch_size]
        Returns:
            logits: 预测类别 [batch_size, num_classes]
        """
        xrd_seq, xrd_mask = self.xrd_encoder(peaks_x, peaks_y, peaks_mask) # 形状为 [batch_size, peak_max_len+1, embedding_dim]
        xrd_embedding = self.transformer(xrd_seq, src_key_padding_mask=xrd_mask) # 形状为 [batch_size, peak_max_len+1, embedding_dim]
        xrd_embedding = xrd_embedding[:, 0, :]  # 取CLS或第一个token。丢了其它token的信息。形状为 [batch_size, embedding_dim]

        formula_embedding = self.formula_embedding(formula)
        emb_dim = formula_embedding.shape[-1]
        formula_mask = torch.logical_not(formula_mask.unsqueeze(-1).repeat(1, 1, emb_dim)).to(formula_embedding.dtype)
        formula_embedding = torch.mul(formula_embedding, formula_mask)
        formula_embedding = torch.sum(formula_embedding, dim=1) # 形状为 [batch_size, emb_dim]

        # 处理这三个特征
        # 假设 space_group, bandgap 都是一维 [batch]
        # lattice_parameter: [batch, 6]
        extra_features = torch.cat([            
            space_group.unsqueeze(1),        # [batch, 1]。会把 shape 从 [batch] 变成 [batch, 1]，即每个样本的类别编号变成一列
            bandgap.unsqueeze(1),           # [batch, 1]
            density.unsqueeze(1)            # [batch, 1]
        ], dim=1)  # [batch, 3]

        # embedding = torch.cat([xrd_embedding, formula_embedding], dim=-1)
        # 拼接到 embedding
        embedding = torch.cat([xrd_embedding, formula_embedding, extra_features], dim=-1)

        logits1 = self.mlp1(embedding)
        logits2 = self.mlp2(embedding)
        return logits1, logits2


if __name__ == '__main__':
    model = XRDClassificationRegressionModel()
    peaks_x = torch.randn(32, 50) # 32个样本，每个样本50个峰值，数据类型为float32
    peaks_y = torch.randn(32, 50)
    peaks_mask = torch.ones(32, 50)
    formula = torch.randint(0, 118, (32, 10)) # 取值范围是0~117。32个样本，每个样本10个元素。
    formula_mask = torch.zeros(32, 10) # 32个样本，每个样本10个元素。mask全0表示都有效
    lattice_parameter = torch.randn(32, 6) # 32个样本，每个样本6个晶格参数，数据类型为float32
    space_group = torch.randint(1, 230, (32,))# 32个样本，空间群取值范围是1~229
    bandgap = torch.randn(32,)
    density = torch.randn(32,)
    logits1, logits2 = model(peaks_x, peaks_y, peaks_mask, formula, formula_mask, space_group, bandgap, density)
    print(logits1.shape)
    print(logits2.shape)
