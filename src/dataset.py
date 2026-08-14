"""
VizWiz dataset and dataloader utilities.
Handles annotation loading, image path resolution, and tokenization.
"""

import collections
import json
import os
from typing import List, Tuple, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2Tokenizer


class VizWizDataset(Dataset):
    def __init__(self, img_paths: List[str], captions: List[str], preprocess, tokenizer, max_length: int = 50):
        self.img_paths = img_paths
        self.captions = captions
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.preprocess(image)
        tokens = self.tokenizer(
            self.captions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return image, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)


def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)
    return torch.stack(images), torch.stack(input_ids), torch.stack(attention_masks)


def load_split(
    annotation_path: str,
    images_dir: str,
    tokenizer: GPT2Tokenizer,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    with open(annotation_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["annotations"])
    if limit:
        df = df.iloc[:limit]

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    image_path_to_caption = collections.defaultdict(list)
    for _, row in df.iterrows():
        fname    = id_to_filename[row["image_id"]]
        img_path = os.path.join(images_dir, fname)
        caption  = f"{tokenizer.bos_token} {row['caption']} {tokenizer.eos_token}"
        if len(caption) <= 300:
            image_path_to_caption[img_path].append(caption)

    img_paths, captions = [], []
    for path, caps in image_path_to_caption.items():
        captions.extend(caps)
        img_paths.extend([path] * len(caps))

    print(f"  Loaded {len(captions)} captions over {len(image_path_to_caption)} unique images")
    return img_paths, captions


def filter_existing(img_paths: List[str], captions: List[str]) -> Tuple[List[str], List[str]]:
    """Drop entries whose image file is missing from disk (e.g. incomplete Drive sync)."""
    kept_paths, kept_captions, missing = [], [], 0
    for p, c in zip(img_paths, captions):
        if os.path.exists(p):
            kept_paths.append(p)
            kept_captions.append(c)
        else:
            missing += 1
    if missing:
        print(f"  Warning: {missing} image file(s) not found — skipped.")
    return kept_paths, kept_captions


def build_dataloaders(
    annotations_path: dict,
    images_path: dict,
    preprocess,
    tokenizer: GPT2Tokenizer,
    batch_size: int = 16,
    train_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, VizWizDataset, VizWizDataset]:
    print("Loading train split...")
    train_imgs, train_caps = load_split(
        annotations_path["train"], images_path["train"], tokenizer, limit=train_limit
    )
    train_imgs, train_caps = filter_existing(train_imgs, train_caps)

    print("Loading val split...")
    val_imgs, val_caps = load_split(
        annotations_path["val"], images_path["val"], tokenizer
    )
    val_imgs, val_caps = filter_existing(val_imgs, val_caps)

    train_ds = VizWizDataset(train_imgs, train_caps, preprocess, tokenizer)
    val_ds = VizWizDataset(val_imgs, val_caps, preprocess, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dl, val_dl, train_ds, val_ds