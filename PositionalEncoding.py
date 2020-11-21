import math

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len, 1], λ�ñ���
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # ����ά��
        # print(pe.shape)
        self.register_buffer('pe', pe)  # �ڴ��ж�һ��������ģ�ͱ���ͼ��ص�ʱ�򣬿���д��Ͷ���

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # Embedding + PositionalEncoding
        return self.dropout(x)