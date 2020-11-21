# coding=gbk
import torchsnooper
from torch import nn

from InputEmbeddings import InputEmbeddings
from MultiHeadAttention import MultiHeadedAttention
from PositionalEncoding import PositionalEncoding


class MyTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, p_drop, h, output_size):
        super(MyTransformerModel, self).__init__()
        self.drop = nn.Dropout(p_drop)
        # Embeddings,
        self.embeddings = InputEmbeddings(vocab_size=vocab_size, embedding_dim=embedding_dim)
        # H: [e_x1 + p_1, e_x2 + p_2, ....]
        self.position = PositionalEncoding(embedding_dim, p_drop)
        # Multi-Head Attention
        self.atten = MultiHeadedAttention(h, embedding_dim)  # self-attention-->����һ��ȫ���ӵ�����ṹ
        # ���һ��(LayerNorm)
        self.norm = nn.LayerNorm(embedding_dim)
        # Feed Forward
        self.linear = nn.Linear(embedding_dim, output_size)
        # ��ʼ������
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def forward(self, inputs, mask):  # ά�Ⱦ�Ϊ: [batch, seq_len]

        embeded = self.embeddings(inputs)  # 1. InputEmbedding [batch, seq_len, embedding_dim]
        # print("embedingshape:",embeded.shape)              # torch.Size([36, 104, 100])

        embeded = self.position(embeded)  # 2. PosionalEncoding [batch, seq_len, embedding_dim]
        #         print(embeded.shape)              # torch.Size([36, 104, 100])

        mask = mask.unsqueeze(2)  # [batch, seq_len, 1]

        # 3.1 MultiHeadedAttention [batch, seq_len. embedding_dim]
        inp_atten = self.atten(embeded, embeded, embeded, mask)
        # 3.2 LayerNorm [batch, seq_len, embedding_dim]
        inp_atten = self.norm(inp_atten + embeded)
        #         print(inp_atten.shape)            # torch.Size([36, 104, 100])

        # 4. Masked, [batch, seq_len, embedding_dim]
        inp_atten = inp_atten * mask  # torch.Size([36, 104, 100])

        #         print(inp_atten.sum(1).shape, mask.sum(1).shape)  # [batch, emb_dim], [batch, 1]
        b_avg = inp_atten.sum(1) / (mask.sum(1) + 1e-5)  # [batch, embedding_dim]

        return self.linear(b_avg).squeeze()  # [batch, 1] -> [batch]