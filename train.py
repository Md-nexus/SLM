print("Running imports...\n")
import gc
import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model.slm import SLM
import torch.optim as optim

print("All imports completed\n")

print("Preparing Environment...\n")
# ---CLEAR CACHE--- #
torch.cuda.empty_cache()
gc.collect()

# ---MAX TREADS--- #
torch.set_num_threads(torch.get_num_threads())

# ---GET FILES--- #
print("Getting necessary files--->  <---\n")
with open("data/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

stoi = vocab["stoi"]
itos = {int(i): ch for i, ch in vocab["itos"].items()}
vocab_size = len(stoi)

with open("data/encoded.txt", "r", encoding="utf-8") as f:
    data = list(map(int, f.read().split()))
print("Files loaded\n")

print("Creating Batches...\n")


def create_batches(data, seq_length, batch_size):
    num_batches = len(data) // (seq_length * batch_size)
    data = data[: num_batches * seq_length * batch_size]
    x = np.array(data)
    y = np.roll(x, -1)
    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)

    x_batches = []
    y_batches = []

    for i in range(0, x.shape[1], seq_length):
        x_batches.append(x[:, i:i + seq_length])
        y_batches.append(y[:, i:i + seq_length])

    return torch.tensor(np.array(x_batches)), torch.tensor(np.array(y_batches))


print("Created Batches\n")

# ---CONFIG--- #
seq_length = 64
batch_size = 32
num_epochs = int(input("Set number of epochs: "))
learning_rate = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---DATA PREP--- #
x_batches, y_batches = create_batches(data, seq_length, batch_size)
print(f"Total batches: {x_batches.shape[0]}")

# ---BUILD MODEL--- #
model = SLM(vocab_size, embed_size=128, hidden_size=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---LOAD WEIGHTS IF THEY EXIST--- #
weights_path = "model/slm_weight.pt"
if os.path.exists(weights_path):
    print("Loading existing weights for fine-tuning... (^.^)")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    learning_rate = 0.00005
    # --RE-INSTANTIATE optimizer-- #
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

else:
    print("No exisxting weights found (X_X).\n" "Training from scratch (-_-)")

print("Starting training...\n")
# ---TRAINING LOOP--- #
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(range(x_batches.shape[0]), desc=f"Epoch {epoch+1}/{num_epochs}"):
        x = x_batches[batch].to(device)
        y = y_batches[batch].to(device)

        optimizer.zero_grad()
        output, _ = model(x)

        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / x_batches.shape[0]
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
print("Saving Model")


# ---SAVE TRAINED MODEL--- #
torch.save(model.state_dict(), "model/slm_weight.pt")
print("Training complete.")

# ---END--- #
