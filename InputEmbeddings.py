import math

from torch import nn


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embedding_dim)