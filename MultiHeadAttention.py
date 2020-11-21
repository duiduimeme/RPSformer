import copy
import math
import torch.nn.functional as F
import torch
from torch import nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):  # q,k,v: [batch, h, seq_len, d_k]
    d_k = query.size(-1)  # query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 打分机制 [batch, h, seq_len, seq_len]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # mask==0的内容填充-1e9, 使计算softmax时概率接近0
    p_atten = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分, [batch, h, seq_len, seq_len]

    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten  # [batch, h, seq_len, d_k]


# 建立一个全连接的网络结构
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % h == 0

        self.d_k = embedding_dim // h  # 将 embedding_dim 分割成 h份 后的维度
        self.h = h  # h 指的是 head数量
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # q,k,v: [batch, seq_len, embedding_dim]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, seq_len, 1]
        nbatches = query.size(0)

        # 1. Do all the linear projections(线性预测) in batch from embeddding_dim => h x d_k
        # [batch, seq_len, h, d_k] -> [batch, h, seq_len, d_k]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2. Apply attention on all the projected vectors in batch.
        # atten:[batch, h, seq_len, d_k], p_atten: [batch, h, seq_len, seq_len]
        attn, p_atten = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. "Concat" using a view and apply a final linear.
        # [batch, h, seq_len, d_k]->[batch, seq_len, embedding_dim]
        attn = attn.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](attn)