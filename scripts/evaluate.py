import os, json, re, random, collections
import torch
import clip
import easyocr
import zipfile
import time
from typing import Optional
from PIL import Image
from tqdm import tqdm

from transformers import GPT2Tokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from src.model import ClipCaptionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ── Paths — edit these ────────────────────────────────────────────────────────
VIZWIZ_ROOT     = "/content/drive/MyDrive/vizwiz"
UNZIP_ROOT      = "/content/vizwiz"
CHECKPOINT_PATH = "/content/drive/MyDrive/clipcap_checkpoints_new/model_checkpoint_best.pth"
TOKENIZER_DIR   = "/content/drive/MyDrive/clipcap_checkpoints_new/tokenizer"
PREFIX_LENGTH   = 10
EVAL_N          = 300
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(UNZIP_ROOT, exist_ok=True)
val_dest = os.path.join(UNZIP_ROOT, "val")
val_zip  = os.path.join(VIZWIZ_ROOT, "val.zip")

if os.path.exists(val_dest) and len(os.listdir(val_dest)) > 0:
    print(f"val: already extracted ({len(os.listdir(val_dest))} files)")
else:
    print("val: extracting...", end=" ", flush=True)
    t = time.time()
    with zipfile.ZipFile(val_zip, 'r') as z:
        z.extractall(UNZIP_ROOT)
    print(f"done in {time.time()-t:.0f}s")

images_val_dir       = val_dest
annotations_val_path = os.path.join(VIZWIZ_ROOT, "annotations", "val.json")

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

_ocr_reader = None

def get_ocr_text(image_path, max_words=5):
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    try:
        results = _ocr_reader.readtext(image_path)
    except Exception as e:
        print(f"  [OCR warning] skipped {os.path.basename(image_path)}: {e}")
        return ""
    raw   = " ".join(r[1] for r in results)
    words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", "", raw).split() if len(w) > 2]
    return " ".join(words[:max_words])


def load_model(checkpoint_path, tokenizer_dir, prefix_length=PREFIX_LENGTH):
    tokenizer  = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    model      = ClipCaptionModel(prefix_length=prefix_length, vocab_size=len(tokenizer)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer

model, tokenizer = load_model(CHECKPOINT_PATH, TOKENIZER_DIR)
print("Model loaded.")

with open(annotations_val_path, 'r') as f:
    val_data = json.load(f)

id_to_filename = {img['id']: img['file_name'] for img in val_data['images']}

gts = collections.defaultdict(list)
for ann in val_data['annotations']:
    if ann.get('is_rejected', False):
        continue
    gts[ann['image_id']].append(ann['caption'])

val_image_ids = [iid for iid, caps in gts.items() if len(caps) > 0]
print(f"{len(val_image_ids)} val images with reference captions available.")

random.seed(0)
eval_ids = random.sample(val_image_ids, min(EVAL_N, len(val_image_ids))) if EVAL_N else val_image_ids


def generate_captions(model, tokenizer, image_ids, images_dir, id_to_filename,
                      ocr_cache=None, max_length=50):
    results = {}
    for iid in tqdm(image_ids, desc="Generating"):
        img_path = os.path.join(images_dir, id_to_filename[iid])
        if not os.path.exists(img_path):
            continue

        ocr_text = ocr_cache.get(iid, "") if ocr_cache else get_ocr_text(img_path)

        image      = Image.open(img_path).convert("RGB")
        clip_input = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            clip_features = clip_model.encode_image(clip_input).float()
            prefix        = model.clip_project(clip_features).view(1, model.prefix_length, model.gpt_embedding_size)

            start_id = tokenizer.bos_token_id
            end_id   = tokenizer.eos_token_id
            tokens   = [start_id]
            if ocr_text:
                tokens.extend(tokenizer.encode(ocr_text, add_special_tokens=False))
            tokens = torch.tensor([tokens], device=device)

            for _ in range(max_length):
                outputs    = model.gpt(inputs_embeds=torch.cat((prefix, model.gpt.transformer.wte(tokens)), dim=1))
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                if next_token.item() == end_id:
                    break
                tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

        caption = tokenizer.decode(tokens.squeeze().tolist(), skip_special_tokens=True)
        caption = caption.replace("<start>", "").split("<end>")[0].strip()
        results[iid] = caption

    return results


ocr_cache = {}
for iid in tqdm(eval_ids, desc="OCR"):
    ocr_cache[iid] = get_ocr_text(os.path.join(images_val_dir, id_to_filename[iid]))

captions = generate_captions(model, tokenizer, eval_ids, images_val_dir, id_to_filename, ocr_cache)


def sanitize(res):
    return {
        iid: [c.replace("\n", " ").replace("\r", " ").strip() or "." for c in caps]
        for iid, caps in res.items()
    }


def score_captions(gts_subset, res):
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(),  "ROUGE-L"),
        (Cider(),  "CIDEr"),
    ]
    scores = {}
    for scorer, names in scorers:
        try:
            score, _ = scorer.compute_score(gts_subset, res)
        except Exception as e:
            label = names if isinstance(names, str) else names[0]
            print(f"  [warning] {label} scorer failed: {e}")
            continue
        if isinstance(names, list):
            for n, s in zip(names, score):
                scores[n] = s
        else:
            scores[names] = score
    return scores


gts_subset = {iid: gts[iid] for iid in captions}
res        = sanitize({iid: [captions[iid]] for iid in captions})

print(f"\n=== MODEL EVALUATION (n={len(captions)}) ===")
scores = score_captions(gts_subset, res)
for k, v in scores.items():
    print(f"  {k}: {v:.4f}")


def repetition_rate(captions_dict, n=4):
    repeated = 0
    for cap in captions_dict.values():
        words  = cap.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if len(ngrams) != len(set(ngrams)):
            repeated += 1
    return repeated / len(captions_dict)


def ocr_recall(captions_dict, ocr_cache):
    checked, hit = 0, 0
    for iid, cap in captions_dict.items():
        ocr_text = ocr_cache.get(iid, "")
        if not ocr_text:
            continue
        checked += 1
        if set(ocr_text.lower().split()) & set(cap.lower().split()):
            hit += 1
    return hit / checked if checked else None


print(f"\nRepetition rate : {repetition_rate(captions):.4f}")
print(f"OCR recall      : {ocr_recall(captions, ocr_cache):.4f}")

print("\n=== Sample captions ===")
for iid in random.sample(list(captions.keys()), min(8, len(captions))):
    print(f"\nImage : {id_to_filename[iid]}")
    print(f"  Ref : {gts[iid][0]}")
    print(f"  Gen : {captions[iid]}")