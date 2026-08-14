"""
Single-image inference: CLIP encode → prefix project → GPT-2 greedy decode.
OCR text is injected into the token stream at inference time.
"""

import torch
import easyocr
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, GPT2Tokenizer

from src.model import ClipCaptionModel


def get_ocr_text(image_path: str, max_words: int = 5) -> str:
    reader = easyocr.Reader(["en"])
    results = reader.readtext(image_path)
    words = " ".join(res[1] for res in results).split()
    filtered = [w for w in words if len(w) > 2 and not w.isnumeric()]
    return " ".join(filtered[:max_words]).strip()


def generate_caption(
    image_path: str,
    model: ClipCaptionModel,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    tokenizer: GPT2Tokenizer,
    ocr_text: str,
    device: torch.device,
    max_length: int = 50,
) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        clip_features = clip_model.get_image_features(pixel_values).float()

    tokens = [tokenizer.bos_token_id]
    if ocr_text:
        tokens.extend(tokenizer.encode(ocr_text, add_special_tokens=False))
    tokens = torch.tensor([tokens], device=device)

    with torch.no_grad():
        prefix = model.clip_project(clip_features).view(1, model.prefix_length, model.gpt_embedding_size)
        for _ in range(max_length):
            token_embeds = model.gpt.transformer.wte(tokens)
            outputs = model.gpt(inputs_embeds=torch.cat((prefix, token_embeds), dim=1))
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

    caption = tokenizer.decode(tokens.squeeze().tolist(), skip_special_tokens=True)
    return caption.replace("<start>", "").split("<end>")[0].strip()


def load_model_for_inference(
    checkpoint_path: str,
    tokenizer_dir: str,
    prefix_length: int = 10,
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
    model = ClipCaptionModel(prefix_length=prefix_length, vocab_size=len(tokenizer)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, clip_model, clip_processor, tokenizer, device