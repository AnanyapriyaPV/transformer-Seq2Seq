import torch
import torch.nn as nn
from attention_masks import generate_causal_mask

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layer = nn.TransformerDecoderLayer(embed_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        tgt_mask = generate_causal_mask(tgt.size(0)).to(tgt.device)
        tgt = self.embed(tgt)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.fc(out)
