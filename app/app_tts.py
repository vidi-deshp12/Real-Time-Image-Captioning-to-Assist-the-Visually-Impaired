from flask import Flask, request, jsonify, send_file
from gtts import gTTS
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer
from model import ClipCaptionModel
from huggingface_hub import hf_hub_download
import easyocr
import re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
TTS_FOLDER = "tts_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TTS_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "vidi-deshp/img-caption-model"

# Load model weights
checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
model = ClipCaptionModel(prefix_length=10).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = GPT2Tokenizer.from_pretrained(REPO_ID)

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

def get_ocr_text(image_path):
    """Extracts text from an image using EasyOCR."""
    results = reader.readtext(image_path)
    extracted_text = " ".join([res[1] for res in results])  # Combine detected words

    # Clean text (remove single characters, numbers, etc.)
    words = extracted_text.split()
    filtered_words = [word for word in words if len(word) > 2 and not word.isnumeric()]
    return " ".join(filtered_words[:5]).strip()  # Limit to 5 words

def clean_ocr_text(ocr_text):
    """Cleans OCR text by removing symbols and extra spaces."""
    ocr_text = re.sub(r"[^a-zA-Z0-9\s]", "", ocr_text)  # Remove special characters
    words = ocr_text.split()
    return " ".join(words[:5]).strip()  # Keep first 5 words

def generate_caption(image_path):
    """Generate an image caption using CLIP-GPT2 with OCR conditioning."""
    image = Image.open(image_path).convert("RGB")

    #  Extract CLIP features
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**inputs).float()

    #  Extract & clean OCR text
    ocr_text = get_ocr_text(image_path)
    clean_text = clean_ocr_text(ocr_text)
    print(f"[OCR]: {clean_text}")

    #  **Generate prompt**
    if clean_text:
        prompt = f"Text detected: {clean_text}. Describe the scene:"
    else:
        prompt = "Describe the scene:"

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prefix = model.clip_project(clip_features).view(1, model.prefix_length, model.gpt_embedding_size)

    max_length = 50
    with torch.no_grad():
        for _ in range(max_length):
            gpt2_embeddings = model.gpt.transformer.wte(input_ids)
            inputs_embeds = torch.cat((prefix, gpt2_embeddings), dim=1)
            outputs = model.gpt(inputs_embeds=inputs_embeds)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)

    #  Decode caption & clean up
    caption = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)

    # Remove repetition
    unique_sentences = list(dict.fromkeys(caption.split(". ")))
    caption = ". ".join(unique_sentences).strip()

    #  **Remove unnecessary text**
    caption = caption.replace("Describe the scene:", "").strip()
    caption = caption.replace("Text detected:", "").strip()
    caption = caption.replace("<start>", "").strip()
    caption = caption.split("<end>")[0].strip()  # Stop at first `<end>`

    # Append cleaned OCR text only if it's not already in the caption
    if clean_text.lower() not in caption.lower():
        final_caption = f"{caption}. (Detected text: {clean_text})"
    else:
        final_caption = caption

    return final_caption

def text_to_speech(text):
    """Convert text to speech and save as an audio file."""
    tts = gTTS(text=text, lang="en")
    filename = "caption_audio.mp3"
    filepath = os.path.join(TTS_FOLDER, filename)
    tts.save(filepath)
    return filename

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)


    caption = generate_caption(file_path)
    audio_filename = text_to_speech(caption)


    #http://192.168.129.204
    #  Return complete backend URL for audio
    return jsonify({
        "caption": caption,
        "audio_url": f"http://192.168.7.204:5000/tts_audio/{audio_filename}"
    })

@app.route('/tts_audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serve generated audio file."""
    filepath = os.path.join(TTS_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="audio/mp3")
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
