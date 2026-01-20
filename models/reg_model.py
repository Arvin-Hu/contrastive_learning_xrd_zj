import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
from torch import Tensor
from models.qformer import QFormer
from models.model import PeakTokenizer, XRDInputEncoder


    

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
    
    

class XRDFormulaModel(nn.Module):
    """
    XRD回归模型，输入XRD图谱和化学式原子，预测属性
    """
    def __init__(self,                 
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.0,
                 formula_vocab_size: int = 118
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
            nn.Linear(2 * embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.formula_embedding = nn.Embedding(formula_vocab_size, embedding_dim)
        
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
    
    def forward(self, peaks_x: torch.Tensor, peaks_y: torch.Tensor, peaks_mask: torch.Tensor, formula: torch.Tensor, formula_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            peaks_x: XRD图谱峰值的角度 [batch_size, seq_len]
            peaks_y: XRD图谱峰值的强度 [batch_size, seq_len]
            peaks_mask: XRD输入数据的掩码 [batch_size, seq_len]
            formula: 化学式 [batch_size, formula_seq_len]
            formula_mask: 化学式掩码 [batch_size, formula_seq_len]
        Returns:
            logits: 预测值 [batch_size, 1]
        """
        # xrd_embedding = self.xrd_encoder(xrd_input)
        xrd_seq, xrd_mask = self.xrd_encoder(peaks_x, peaks_y, peaks_mask)
        xrd_embedding = self.transformer(xrd_seq, src_key_padding_mask=xrd_mask)
        xrd_embedding = xrd_embedding[:, 0, :]
        formula_embedding = self.formula_embedding(formula)
        emb_dim = formula_embedding.shape[-1]
        formula_mask = torch.logical_not(formula_mask.unsqueeze(-1).repeat(1, 1, emb_dim)).to(formula_embedding.dtype)
        # print(formula_embedding.shape)
        # print(formula_mask.shape)
        formula_embedding = torch.mul(formula_embedding , formula_mask)
        formula_embedding = torch.sum(formula_embedding, dim=1)
        embedding = torch.cat([xrd_embedding, formula_embedding], dim=-1)
        logits = self.mlp(embedding)
        
        return logits


class XRDConvRegressionModel(nn.Module):
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
        
        self.xrd_encoder = XRDInputEncoder(input_dim=3751,d_model=embedding_dim)
        
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
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: XRD图谱强度数组 [batch_size, seq_len]
        Returns:
            logits: 预测值 [batch_size, 1]
        """
        # xrd_embedding = self.xrd_encoder(xrd_input)
        x = self.xrd_encoder(x)
        x = self.transformer(x)
        x = x[:, 0, :]
        
        logits = self.mlp(x)
        
        return logits
    
    
if __name__ == '__main__':
    model = XRDConvRegressionModel()
    peaks_x = torch.randn(32, 50)
    peaks_y = torch.randn(32, 50)
    peaks_mask = torch.ones(32, 50)
    mask = torch.ones(32, 64)
    inputs = torch.randn(32, 3751)
    logits = model(inputs)
    print(logits.shape)
    