import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerEncoder
# torch.nn.modules.activation




class LearnablePosEncoding(nn.Module):
    """可学习绝对位置编码，支持 2D/3D... 任意 shape，最后一维是 dim."""
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))   # [1, L, D]
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        # x: [B, L, D]
        return x + self.pos_emb[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        # print(batch_size, seq_len, attn_mask)
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # print("\t", Q.shape, K.shape, V.shape)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # print("\t", scores.shape, torch.max(scores), torch.min(scores), scores.dtype)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len]
            # Reshape to match attention scores: [batch_size, 1, 1, seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # print("\t", K.shape, torch.max(K), torch.min(K), "\n")
            # print("\t", Q.shape, torch.max(Q), torch.min(Q), "\n")
            # print("\t", V.shape, torch.max(V), torch.min(V), "\n")
            # print("\t", scores.shape, torch.max(scores), torch.min(scores), "\n")
            # print("\t", mask.shape, torch.max(mask), torch.min(mask))
            scores = scores.masked_fill(mask == 0, -1e9)
            # print("\t", scores.shape, mask.shape, key.shape, torch.max(scores), torch.min(scores), scores.dtype)
            # print("\t", scores[2, 1, 4, :])
            # print("\t", )
            # print(mask)
        
        attention_weights = F.softmax(scores, dim=-1)
        # print("\t", attention_weights[2, 1, 4, :])
        # print("\t", attention_weights.shape, torch.max(attention_weights), torch.min(attention_weights), "\n")
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        return self.out_linear(context), attention_weights

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        
    def forward(self, x, key_padding_mask=None, attn_mask=None):

        x = self.norm1(x)
        attn_output, attn_weights = self.attention(x, x, x, key_padding_mask, attn_mask)
        x = x + attn_output
        
        return x, attn_weights

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, context, context_mask=None):
        # Cross-attention: queries attend to context
        # tmp_mask = torch.tensor(context_mask, dtype=torch.long)
        # print("\t", context_mask.shape, torch.max(tmp_mask), torch.min(tmp_mask))
        queries = self.norm1(queries)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            queries, context, context, context_mask
        )
        queries = queries + cross_attn_output
        
        # Feed-forward
        queries = self.norm2(queries)
        ff_output = self.ff(queries)
        queries = queries + self.dropout(ff_output)
        
        return queries, cross_attn_weights

class QFormer(nn.Module):
    def __init__(self, 
                 input_dim=64,
                 output_dim=1024,
                 mid_dim=512,  # 新增中间层维度
                 num_queries=4,
                 num_layers=6,
                 num_heads=8,
                 ff_dim=2048,
                 dropout=0.0):
        super().__init__()
        
        self.num_queries = num_queries
        self.output_dim = output_dim
        self.mid_dim = mid_dim  # 保存中间层维度
        
        # Learnable query tokens - 使用mid_dim作为中间维度
        # self.query_tokens = nn.Parameter(torch.randn(1, num_queries, mid_dim))
        self.query_tokens = nn.Embedding(num_queries, mid_dim)
        # print(torch.max(self.query_tokens), torch.min(self.query_tokens))
        
        # 输入投影到中间维度，而不是直接到输出维度
        # print("----------------", input_dim, type(input_dim), mid_dim, type(mid_dim))
        self.input_proj = nn.Linear(input_dim, mid_dim)
        
        # Cross-attention layers for interaction between queries and input
        # 使用mid_dim作为中间维度
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(mid_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Self-attention layers for query interactions
        # 使用mid_dim作为中间维度
        self.self_attention_layers = nn.ModuleList([
            TransformerBlock(mid_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 中间层归一化
        self.mid_norm = nn.LayerNorm(mid_dim)
        
        # 新增：从中间维度映射到输出维度的投影层
        self.output_proj = nn.Linear(mid_dim, output_dim)
        
        # Output normalization - 现在在输出维度上
        self.output_norm = nn.LayerNorm(output_dim)
        
        # 1. CIF 序列的位置编码
        # self.cif_pos_enc = LearnablePosEncoding(ff_dim, mid_dim)

        # 2. Query tokens 的位置编码
        # self.query_pos_enc = LearnablePosEncoding(num_queries, mid_dim)
        
        self.apply(self._init_weights)
        # print(torch.max(self.query_tokens), torch.min(self.query_tokens))
        
    def _init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            # print(type(module))
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            # Initialize query tokens
            # print(type(module))
            nn.init.normal_(module, mean=0.0, std=0.02)
            
    def forward(self, x, key_padding_mask, lengths=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            lengths: Optional tensor of shape [batch_size] indicating actual sequence lengths
                   If None, assumes all sequences are full length
        
        Returns:
            Output tensor of shape [batch_size, num_queries, output_dim]
        """

        batch_size, seq_len, input_dim = x.shape
        
        x = torch.Tensor(x).to(self.input_proj.weight.dtype)
        cif_features = self.input_proj(x)  # [batch_size, seq_len, mid_dim]
        # print(x_proj.device, x.device, self.input_proj.weight.device)
        # cif_features = self.cif_pos_enc(cif_features)  # ← 新增
        
        # 扩展查询tokens到batch size - 现在是mid_dim维度
        # print(self.query_tokens.weight.shape)
        # print(torch.max(self.query_tokens.weight))
        queries = self.query_tokens.weight.reshape(1, -1, self.mid_dim)
        queries = queries.expand(batch_size, -1, -1)  # [batch_size, num_queries, mid_dim]
        # queries = self.query_pos_enc(queries)      
        
        # Store attention weights for analysis
        cross_attn_weights_list = []
        self_attn_weights_list = []
        
        # Alternate between cross-attention and self-attention
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            
            cif_features, self_attn_weights = self_attn(cif_features, key_padding_mask)
            self_attn_weights_list.append(self_attn_weights)

            queries, cross_attn_weights = cross_attn(queries, cif_features, key_padding_mask)
            cross_attn_weights_list.append(cross_attn_weights)

        
        # 新增：从中间维度映射到输出维度
        output = self.output_proj(queries)
        output = torch.sum(output, dim=1)
        
        return output