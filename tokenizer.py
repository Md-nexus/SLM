import json

with open("data/brain.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(l):
    return "".join(itos[i] for i in l)


data_encoded = encode(text)
with open("data/vocab.json", "w", encoding="utf-8") as f:
    json.dump({"stoi": stoi, "itos": itos}, f, indent=4, ensure_ascii=False)

with open("data/encoded.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, data_encoded)))

print("Tokenization complete.")
