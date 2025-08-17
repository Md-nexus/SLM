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
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import random

print("All imports completed\n")

print("Preparing Environment...\n")
# ---CLEAR CACHE--- #
torch.cuda.empty_cache()
gc.collect()

# ---MAX THREADS--- #
torch.set_num_threads(torch.get_num_threads())

# ---LOAD TOKENIZER--- #
print("Loading tokenizer...\n")
tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

# ---GET ENCODED DATA--- #
print("Loading encoded data...\n")
with open("data/encoded.txt", "r", encoding="utf-8") as f:
    data = list(map(int, f.read().split()))
print(f"Total tokens: {len(data)}")

print("Creating Batches...\n")

def create_batches(data, seq_length, batch_size, train_split=0.9):
    """Create training and validation batches with better data handling"""
    # Split data into train and validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    def process_data(data_subset):
        num_batches = len(data_subset) // (seq_length * batch_size)
        data_subset = data_subset[: num_batches * seq_length * batch_size]
        x = np.array(data_subset)
        y = np.roll(x, -1)
        y[-1] = x[0]  # Wrap around for the last token
        
        x = x.reshape(batch_size, -1)
        y = y.reshape(batch_size, -1)

        x_batches = []
        y_batches = []

        for i in range(0, x.shape[1], seq_length):
            if i + seq_length <= x.shape[1]:
                x_batches.append(x[:, i:i + seq_length])
                y_batches.append(y[:, i:i + seq_length])

        return torch.tensor(np.array(x_batches)), torch.tensor(np.array(y_batches))
    
    train_x, train_y = process_data(train_data)
    val_x, val_y = process_data(val_data)
    
    return train_x, train_y, val_x, val_y

print("Created Batches\n")

# ---CONFIG--- #
seq_length = 128  # Increased context window
batch_size = 16   # Reduced for larger model
num_epochs = int(input("Set number of epochs: "))
learning_rate = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---DATA PREP--- #
train_x, train_y, val_x, val_y = create_batches(data, seq_length, batch_size)
print(f"Training batches: {train_x.shape[0]}")
print(f"Validation batches: {val_x.shape[0]}")

# ---BUILD MODEL--- #
model = SLM(
    vocab_size=vocab_size,
    embed_size=256,
    hidden_size=512,
    num_layers=3,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ---LOAD WEIGHTS IF THEY EXIST--- #
weights_path = "model/slm_weight.pt"
if os.path.exists(weights_path):
    print("Loading existing weights for fine-tuning... (^.^)")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    learning_rate = 0.00005
    # --RE-INSTANTIATE optimizer-- #
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
else:
    print("No existing weights found (X_X).\nTraining from scratch (-_-)")

print("Starting training...\n")

# ---TRAINING LOOP--- #
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_loss = 0
    train_batches = list(range(train_x.shape[0]))
    random.shuffle(train_batches)  # Shuffle training order
    
    for batch_idx in tqdm(train_batches, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
        x = train_x[batch_idx].to(device)
        y = train_y[batch_idx].to(device)

        optimizer.zero_grad()
        output, _ = model(x)

        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / train_x.shape[0]
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx in tqdm(range(val_x.shape[0]), desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
            x = val_x[batch_idx].to(device)
            y = val_y[batch_idx].to(device)
            
            output, _ = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / val_x.shape[0]
    
    # Learning rate scheduling
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print("  Saving best model...")
        torch.save(model.state_dict(), "model/slm_weight.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping after {patience} epochs without improvement")
            break

print("Training complete.")
print(f"Best validation loss: {best_val_loss:.4f}")

# ---END--- #
