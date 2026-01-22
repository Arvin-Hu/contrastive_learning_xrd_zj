import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
from torch import Tensor
from models.qformer import QFormer


class XRDInputEncoder(nn.Module):
    """XRD图谱输入编码器"""
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        
        # MLP编码强度特征
        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, d_model//2),  # 每个强度值作为单独特征
            nn.ReLU(),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, d_model//2)
        )
        
        # 角度位置编码（2θ值）
        self.angle_encoder = nn.Embedding(input_dim, d_model//4)
        
        # 峰值检测卷积层
        self.peak_detector = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 4, kernel_size=25, padding=12),
            nn.ReLU()
        )
        
        # 特征融合
        self.feature_fusion = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        
        # 1. 强度特征编码
        intensity_features = self.intensity_encoder(
            x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        )  # [batch_size, seq_len, d_model]
        # print('intensity_features: ', intensity_features.shape)
        
        # 2. 角度编码
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        angle_features = self.angle_encoder(positions)  # [batch_size, seq_len, d_model]
        # print('angel_features: ', angle_features.shape)
        
        # 3. 峰值特征提取
        conv_features = self.peak_detector(
            x.unsqueeze(1)  # [batch_size, 1, seq_len]
        )  # [batch_size, 64, seq_len]
        conv_features = conv_features.permute(0, 2, 1)
        # print('conv_features: ', conv_features.shape)
        
        # 4. 特征融合
        combined = torch.cat([intensity_features, angle_features, conv_features], dim=-1)
        encoded = self.feature_fusion(combined)  # [batch_size, seq_len, d_model]
        encoded = self.layer_norm(encoded)
        
        return encoded
    

class ContrastiveLearningModel(nn.Module):
    """
    基础对比学习模型
    包含编码器、投影头和不同的对比损失
    """
    def __init__(self, 
                 xrd_input_dim: int = 3751,
                 cif_input_dim: int = 64,
                 cif_num_queries: int = 4,
                 embedding_dim: int = 256,
                 projection_dim: int = 64,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 patch_size: int = 25,
                 use_qformer: bool = False,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        
        self.use_qformer = use_qformer
        self.patch_size = patch_size
        
        if self.use_qformer:
            self.cif_encoder = QFormer(input_dim=cif_input_dim,
                                    output_dim=embedding_dim,
                                    mid_dim=embedding_dim // 2,  
                                    num_queries=cif_num_queries,
                                    num_layers=num_layers,
                                    num_heads=num_heads,
                                    ff_dim=embedding_dim // 2,
                                    dropout=0.0)
        # self.xrd_encoder = XRDTokenizer(input_length=xrd_input_dim, patch_size=patch_size, embedding_dim=embedding_dim)
        self.xrd_encoder = PeakTokenizer(embedding_dim=embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 投影头（将embedding映射到对比学习空间）
        self.xrd_projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.cif_projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
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
    
    def forward(self, cif_input: torch.Tensor, cif_mask: torch.Tensor, peaks_x: torch.Tensor, peaks_y: torch.Tensor, peaks_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            cif_input: CIF输入数据 [batch_size, seq_len, cif_input_dim]
            xrd_input: XRD图谱数据 [batch_size, xrd_input_dim]
            mask: CIF输入数据的掩码 [batch_size, seq_len]
        Returns:
            xrd_embedding: 编码后的xrd embedding [batch_size, embedding_dim]
            xrd_projection: xrd投影后的特征 [batch_size, projection_dim]
            cif_projection: cif投影后的特征 [batch_size, projection_dim]
        """
        # xrd_embedding = self.xrd_encoder(xrd_input)
        xrd_seq, xrd_mask = self.xrd_encoder(peaks_x, peaks_y, peaks_mask)
        xrd_embedding = self.transformer(xrd_seq, src_key_padding_mask=xrd_mask)
        xrd_embedding = xrd_embedding[:, 0, :]
        # print('xrd_embedding: ',xrd_embedding.shape)
        xrd_projection = self.xrd_projector(xrd_embedding)
        xrd_projection = F.normalize(xrd_projection, dim=-1)  # L2归一化
        # print('xrd_projection: ', xrd_projection.shape)
        
        if self.use_qformer:
            cif_embedding = self.cif_encoder(cif_input, cif_mask)
            cif_projection = self.cif_projector(cif_embedding)
            cif_projection = F.normalize(cif_projection, dim=-1)
        else:
            cif_mask_expanded = cif_mask.unsqueeze(-1).float()
            cif_projection = torch.sum(cif_input * cif_mask_expanded, dim=1)
            cif_projection = F.normalize(cif_projection, dim=-1)
        # print('cif_embedding: ', cif_embedding.shape)
        # print('cif_projection: ', cif_projection.shape)
        return xrd_embedding, xrd_projection, cif_projection
    
    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """仅获取embedding（用于下游任务）"""
        with torch.no_grad():
            embedding, _, _ = self.forward(x, y)
        return embedding


class ContrastiveLoss:
    """
    对比学习损失函数集合
    """
    @staticmethod
    def infonce_loss(z_i: torch.Tensor, 
                     z_j: torch.Tensor, 
                     temperature: float = 0.07) -> torch.Tensor:
        """
        InfoNCE损失 (NT-Xent)
        Args:
            z_i, z_j: 正样本对的投影特征 [batch_size, projection_dim]
            temperature: 温度系数
        Returns:
            loss: 对比损失
        """
        batch_size = z_i.shape[0]        
        # 计算相似度矩阵
        # sim_matrix = torch.mm(z_i, z_j.t()) / temperature  # [2*batch_size, 2*batch_size]
        logits1 = z_i @ z_j.T / temperature
        logits2 = z_j @ z_i.T / temperature
        
        # 创建标签：对角线上的正样本对
        labels = torch.arange(batch_size, device=z_i.device)
            
        # 计算交叉熵损失
        loss1 = F.cross_entropy(logits1, labels)
        loss2 = F.cross_entropy(logits2, labels)
        loss = (loss1 + loss2) / 2
        return loss
    
    
class PeakTokenizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512
    ):
        super().__init__()
        self.embed_tokens_and_positions = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.vnode_encoder = nn.Embedding(1, embedding_dim) # 一个无实际用途的嵌入层，仅用于参数初始化（可忽略）。
        self.cls_token_param = nn.Parameter(torch.zeros(1, 1, embedding_dim)) # 一个可学习的参数，shape 为 [1, 1, embedding_dim]，作为“CLS token”，用于全局特征聚合。 在自然语言处理中，tokenizer通常会添加以下特殊标记,通常称为：[CLS]、<s>、<bos>、[BOS]、<start>
   
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  
        batch_size = x.shape[0]
        tokens = self.embed_tokens_and_positions(torch.stack([x, y], dim=-1))
        cls_tokens = self.cls_token_param.expand(batch_size, -1, -1) # 形状为 [batch_size, 1, embedding_dim], 将第一维扩展为 batch_size。另外两维度不变。
        # self.cls_token_param 是一个 nn.Parameter，比如 shape [1, 1, embedding_dim]，它是模型的可训练参数。
        # expand(batch_size, -1, -1) 只是创建一个新的视图，让这个参数在 batch 维上“看起来”变成 [batch_size, 1, embedding_dim]，但底层数据还是同一块内存、同一组参数。
        # 梯度和参数更新只针对原始的 self.cls_token_param，不会因为 expand 而复制或增加参数。
        # 可训练参数量不会变化，始终是 1 * 1 * embedding_dim。
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # 形状为 [batch_size, peak_max_len + 1, embedding_dim]
        cls_mask = torch.zeros(batch_size, 1).to(mask.device, dtype=mask.dtype)
        mask = torch.cat([cls_mask, mask], dim=1)
        return tokens, mask


class XRDTokenizer(nn.Module):
    def __init__(
        self,
        input_length: int = 3751,
        patch_size: int = 7,  # 将3751分成约536个patch
        embedding_dim: int = 512
    ):
        """
        XRD衍射数据Tokenizer
        参数:
            input_dim: 输入数据维度，默认为3751
            patch_size: 每个patch包含的数据点数
            hidden_dim: 隐藏层维度（token维度）
            cls_token: 是否添加CLS token
            norm_layer: 归一化层
        """
        super().__init__()
        
        self.input_length = input_length
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        # 计算patch数量
        self.num_patches = input_length // patch_size
        if input_length % patch_size != 0:
            self.num_patches += 1
            
        # 使用1D卷积进行token化（类似Vision Transformer的patch embedding）
        self.projection = nn.Conv1d(
            in_channels=1,  # XRD数据是单通道
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        
        # CLS token
        self.cls_token_param = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            
        # 位置编码
        num_tokens = self.num_patches
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, embedding_dim))
        
        # 归一化层
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入数据，形状为 (batch_size, input_dim) 或 (batch_size, 1, input_dim)
        返回:
            tokens: token化后的数据，形状为 (batch_size, num_tokens, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # 确保输入形状正确
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # 应用投影（token化）
        tokens = self.projection(x)  # (batch_size, hidden_dim, num_patches)
        tokens = tokens.transpose(1, 2)  # (batch_size, num_patches, hidden_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token_param.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 添加位置编码
        tokens = tokens + self.position_embedding
        
        # 归一化
        tokens = self.norm(tokens)
        
        return tokens

# 定义卷积特征提取器
class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_length=3751, embedding_dim=64, patch_size=25):
        super().__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=7, 
            stride=1, 
            padding=3
        )
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        mid_dim = int(input_length / 8)
        self.output_layer = nn.Linear(mid_dim, embedding_dim)
        
    def forward(self, x):
        # x形状: (batch_size, 1, length)
        x = self.relu(self.conv1(x))  # 形状不变
        x = self.pool(x)              # 长度减半
        
        x = self.relu(self.conv2(x))  # 形状不变
        x = self.pool(x)              # 长度减半
        
        x = self.relu(self.conv3(x))  # 形状不变
        x = self.pool(x)              # 长度减半
        
        x = self.output_layer(x)
        x = self.relu(x)
        
        return x


    
    
if __name__ == '__main__':
    model = ContrastiveLearningModel(xrd_input_dim=3751, projection_dim=64)
    cif_input = torch.randn(32, 64, 64)
    # xrd_input = torch.randn(32, 3751)
    peaks_x = torch.randn(32, 50)
    peaks_y = torch.randn(32, 50)
    peaks_mask = torch.ones(32, 50)
    mask = torch.ones(32, 64)
    # xrd_input = xrd_input.unsqueeze(1)
    xrd_embedding, xrd_projection, cif_projection = model(cif_input, mask, peaks_x, peaks_y, peaks_mask)
    print(xrd_embedding.shape)
    
    
    # 使用示例
    # model = ConvFeatureExtractor()
    # data = torch.randn(4, 1, 3751)  # 批量大小=1, 通道=1, 长度=3751
    # features = model(data)  # 提取的特征
    # print(f"输入形状: {data.shape}")
    # print(f"输出特征形状: {features.shape}")
