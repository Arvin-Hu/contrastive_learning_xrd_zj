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




class XRDRegressionModel(nn.Module):
    """
    XRD回归模型，预测属性
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
            nn.Linear(32, 1)
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

class XRDClassificationModel(nn.Module):
    """
    XRD分类模型，预测晶系
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
            nn.Linear(32, 7)
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
            logits: 预测值 [batch_size, 7]
        """
        # xrd_embedding = self.xrd_encoder(xrd_input)
        xrd_seq, xrd_mask = self.xrd_encoder(peaks_x, peaks_y, peaks_mask)
        xrd_embedding = self.transformer(xrd_seq, src_key_padding_mask=xrd_mask)
        xrd_embedding = xrd_embedding[:, 0, :]

        logits = self.mlp(xrd_embedding)

        return logits









if __name__ == '__main__':
    model = XRDRegressionModel()
    peaks_x = torch.randn(32, 50)
    peaks_y = torch.randn(32, 50)
    peaks_mask = torch.ones(32, 50)
    mask = torch.ones(32, 64)
    logits = model(peaks_x, peaks_y, peaks_mask)
    print(logits.shape)
