import torch
import torch.nn as nn
from transformer import Seq2SeqTransformer
import pickle

pairs = [
    ("AI improves healthcare", "AI enhances medical diagnosis <eos>"),
    ("Transformers process data in parallel", "Transformers handle sequences simultaneously <eos>"),
    ("Attention improves NLP accuracy", "Attention mechanisms increase NLP performance <eos>"),
    ("Machine learning helps", "Machine learning helps in data-driven decisions <eos>")
]


# Build vocab

vocab = set()
for s, t in pairs:
    vocab.update(s.split())
    vocab.update(t.split())

vocab = ["<pad>", "<sos>", "<eos>"] + list(vocab)
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

def encode(sentence):
    return [word2idx[w] for w in sentence.split()]

model = Seq2SeqTransformer(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(800):
    total_loss = 0
    for src, tgt in pairs:
        src_ids = torch.tensor(encode(src)).unsqueeze(1)
        tgt_ids = torch.tensor([word2idx["<sos>"]] + encode(tgt)).unsqueeze(1)

        optimizer.zero_grad()
        output = model(src_ids, tgt_ids[:-1])
        loss = criterion(output.view(-1, len(vocab)), tgt_ids[1:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "samples/seq2seq.pth")
print("Training complete.")

with open("samples/vocab.pkl", "wb") as f:
    pickle.dump((word2idx, idx2word), f)
print("Vocabulary saved.")
