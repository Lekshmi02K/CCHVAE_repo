import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pandas as pd
import os
from baseline_model import HAMVAE   # baseline_model.py must be in /content/CCHVAE

# ✅ Paths
TRAIN_PATH = "/content/CCHVAE/processed/train_preprocessed.csv"
VAL_PATH   = "/content/CCHVAE/processed/val_preprocessed.csv"
SAVE_DIR   = "/content/CCHVAE/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Load datasets
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))

# ✅ Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32):   # 🔹 reduced from 64 → 32
        self.texts = df["clean_text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ✅ DataLoaders (batch size doubled)
train_loader = DataLoader(ReviewDataset(train_df, tokenizer), batch_size=32, shuffle=True)
val_loader   = DataLoader(ReviewDataset(val_df, tokenizer), batch_size=32)

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ Model + Optimizer
model = HAMVAE(latent_dim=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# ✅ Mixed precision scaler
scaler = torch.cuda.amp.GradScaler()

# ✅ Training loop
EPOCHS = 2   # 🔹 reduced from 3 → 2
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():   # 🔹 mixed precision
            loss, _ = model(input_ids, attention_mask, labels=input_ids)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if step % 200 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} complete. Avg Train Loss: {avg_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast():
                loss, _ = model(input_ids, attention_mask, labels=input_ids)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"📊 Validation Loss: {avg_val_loss:.4f}")

    # --- Save checkpoint ---
    save_path = os.path.join(SAVE_DIR, f"hamvae_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved: {save_path}")

print("🎯 Training complete!")
