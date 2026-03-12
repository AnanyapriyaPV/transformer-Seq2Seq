import torch
import pickle
from transformer import Seq2SeqTransformer

# Load vocabulary
with open("samples/vocab.pkl", "rb") as f:
    word2idx, idx2word = pickle.load(f)

vocab_size = len(word2idx)

# Load model
model = Seq2SeqTransformer(vocab_size)
model.load_state_dict(torch.load("samples/seq2seq.pth"))
model.eval()

# Encode function
def encode(sentence):
    return [word2idx.get(w, word2idx["<pad>"]) for w in sentence.split()]

# Decode autoregressively
def generate(sentence, max_len=12):
    src = torch.tensor(encode(sentence)).unsqueeze(1)
    memory = model.encoder(src)

    tgt = torch.tensor([word2idx["<sos>"]]).unsqueeze(1)

    for _ in range(max_len):
        out = model.decoder(tgt, memory)
        next_token = out[-1].argmax(dim=-1).item()

        tgt = torch.cat([tgt, torch.tensor([[next_token]])], dim=0)

        if idx2word[next_token] == "<eos>":
            break

    return " ".join(idx2word[i.item()] for i in tgt.squeeze()[1:])

# User input
sentence = input("Enter input: ")
print("Generated output:", generate(sentence))
