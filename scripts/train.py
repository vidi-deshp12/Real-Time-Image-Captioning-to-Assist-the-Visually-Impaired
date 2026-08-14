"""
Training entry point.
Run from repo root: python scripts/train.py
Colab users: set VIZWIZ_ROOT and CHECKPOINT_DIR at the top, then run all cells.
"""

import csv
import os
import random
import time
import zipfile

import clip
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import build_dataloaders, collate_fn
from src.model import ClipCaptionModel
from src.tokenizer_utils import build_tokenizer

# ── Paths — edit these ───────────────────────────────────────────────────────
VIZWIZ_ROOT    = "/content/drive/MyDrive/vizwiz"
CHECKPOINT_DIR = "/content/drive/MyDrive/clipcap_checkpoints_new"
UNZIP_ROOT     = "/content/vizwiz"
# ─────────────────────────────────────────────────────────────────────────────

# Hyperparameters
BATCH_SIZE      = 16
PREFIX_LENGTH   = 10
NUM_EPOCHS      = 5
LR              = 1e-5
WEIGHT_DECAY    = 1e-2
LOG_EVERY       = 10
VAL_EVERY       = 200
VAL_SUBSET_SIZE = 200
PATIENCE        = 3
TRAIN_LIMIT     = 40000


def unzip_splits():
    os.makedirs(UNZIP_ROOT, exist_ok=True)
    for split in ("train", "val"):
        dest = os.path.join(UNZIP_ROOT, split)
        zip_path = os.path.join(VIZWIZ_ROOT, f"{split}.zip")
        if os.path.exists(dest) and len(os.listdir(dest)) > 0:
            print(f"  {split}: already unzipped, skipping.")
            continue
        print(f"  {split}: unzipping...", end=" ", flush=True)
        t = time.time()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(UNZIP_ROOT)
        print(f"done in {time.time()-t:.0f}s")


def save_checkpoint(model, optimizer, epoch, step, loss, path, tokenizer, tokenizer_dir):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "total_loss": loss,
        "vocab_size": len(tokenizer),
    }, path)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"  Saved checkpoint (epoch {epoch}, step {step})")


def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        print("No checkpoint found, starting from scratch.")
        return 0, 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt["epoch"], ckpt["step"], ckpt["total_loss"]


def run_validation(model, clip_model, loader, prefix_length, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, input_ids, masks in loader:
            images, input_ids, masks = images.to(device), input_ids.to(device), masks.to(device)
            features = clip_model.encode_image(images).float()
            prefix_mask = torch.ones(images.size(0), prefix_length, device=device)
            full_mask = torch.cat([prefix_mask, masks], dim=1)
            out = model(tokens=input_ids, prefix=features, mask=full_mask, labels=input_ids)
            total += out.loss.item()
    model.train()
    return total / len(loader)


def run_quick_val(model, clip_model, val_dataset, prefix_length, device, n=VAL_SUBSET_SIZE):
    model.eval()
    idx = random.sample(range(len(val_dataset)), min(n, len(val_dataset)))
    loader = DataLoader(Subset(val_dataset, idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    total = 0.0
    with torch.no_grad():
        for images, input_ids, masks in loader:
            images, input_ids, masks = images.to(device), input_ids.to(device), masks.to(device)
            features = clip_model.encode_image(images).float()
            prefix_mask = torch.ones(images.size(0), prefix_length, device=device)
            full_mask = torch.cat([prefix_mask, masks], dim=1)
            out = model(tokens=input_ids, prefix=features, mask=full_mask, labels=input_ids)
            total += out.loss.item()
    model.train()
    return total / len(loader)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    unzip_splits()

    tokenizer = build_tokenizer()
    vocab_size = len(tokenizer)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    annotations_path = {
        "train": os.path.join(VIZWIZ_ROOT, "annotations", "train.json"),
        "val":   os.path.join(VIZWIZ_ROOT, "annotations", "val.json"),
    }
    images_path = {
        "train": os.path.join(UNZIP_ROOT, "train"),
        "val":   os.path.join(UNZIP_ROOT, "val"),
    }

    train_dl, val_dl, _, val_ds = build_dataloaders(
        annotations_path, images_path, preprocess, tokenizer,
        batch_size=BATCH_SIZE, train_limit=TRAIN_LIMIT,
    )

    model = ClipCaptionModel(prefix_length=PREFIX_LENGTH, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_checkpoint_best.pth")
    tokenizer_dir   = os.path.join(CHECKPOINT_DIR, "tokenizer")
    log_file        = os.path.join(CHECKPOINT_DIR, "training_log.csv")

    start_epoch, _, _ = load_checkpoint(checkpoint_path, model, optimizer, device)

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "loss", "val_loss", "timestamp"])

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        recent_losses = []

        for step, (images, input_ids, masks) in enumerate(train_dl):
            images, input_ids, masks = images.to(device), input_ids.to(device), masks.to(device)

            with torch.no_grad():
                features = clip_model.encode_image(images).float()

            prefix_mask = torch.ones(images.size(0), PREFIX_LENGTH, device=device)
            full_mask = torch.cat([prefix_mask, masks], dim=1)

            out = model(tokens=input_ids, prefix=features, mask=full_mask, labels=input_ids)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            recent_losses.append(loss.item())
            if len(recent_losses) > 50:
                recent_losses.pop(0)

            quick_val = None
            if step > 0 and step % VAL_EVERY == 0:
                quick_val = run_quick_val(model, clip_model, val_ds, PREFIX_LENGTH, device)
                print(f"  [epoch {epoch+1} step {step}] quick val: {quick_val:.4f}")

            if step % LOG_EVERY == 0:
                avg = sum(recent_losses) / len(recent_losses)
                print(f"[epoch {epoch+1} step {step}/{len(train_dl)}] loss={loss.item():.4f}  avg={avg:.4f}")
                with open(log_file, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch + 1, step, f"{loss.item():.6f}",
                        f"{quick_val:.6f}" if quick_val else "",
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    ])

        train_avg = epoch_loss / len(train_dl)
        val_avg = run_validation(model, clip_model, val_dl, PREFIX_LENGTH, device)
        print(f"Epoch {epoch+1}  train={train_avg:.4f}  val={val_avg:.4f}")

        epoch_ckpt = os.path.join(CHECKPOINT_DIR, f"model_checkpoint_epoch{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch + 1, 0, epoch_loss, epoch_ckpt, tokenizer, tokenizer_dir)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch + 1, 0, epoch_loss, checkpoint_path, tokenizer, tokenizer_dir)
            print(f"  → New best val loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()