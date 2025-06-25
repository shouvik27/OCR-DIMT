# translation_engine.py

import torch
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from .segment_utils import jieba_segment

def load_translation_model(model_dir, device):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir).to(device)
    model.eval()
    if device == "cuda":
        model = model.half()
    return tokenizer, model

def translate_batch(texts, tokenizer, model, device, max_length, num_beams):
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **enc, max_length=max_length, num_beams=num_beams, early_stopping=True
        )
    decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
    return decoded

def translate_dict(
    input_dict,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    num_beams,
    segment_func=None,
    progress_desc="Translating"
):
    image_names = list(input_dict.keys())
    src_texts = [input_dict[img] for img in image_names]
    translated = {}
    num_batches = (len(src_texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc=progress_desc):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(src_texts))
        batch_src = src_texts[start:end]
        batch_imgs = image_names[start:end]
        batch_pred = translate_batch(batch_src, tokenizer, model, device, max_length, num_beams)

        for img, pred in zip(batch_imgs, batch_pred):
            seg = segment_func(pred) if segment_func else pred
            translated[img] = seg

    return translated
