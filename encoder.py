import torch
import torch.nn as nn
from attention_masks import generate_causal_mask

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layer = nn.TransformerEncoderLayer(embed_dim, nhead=4)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, x):
        x = self.embed(x)
        return self.encoder(x)
