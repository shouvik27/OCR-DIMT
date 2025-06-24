"""Inference script for LayoutLMv3 + T5 model."""

import os
import json
import torch
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    LayoutLMv3Model,
    T5ForConditionalGeneration,
    AutoTokenizer
)
from torch.nn import LayerNorm, GELU
from dataclasses import dataclass

from ..fine_tune import *

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    test_json: str = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/testset_wo_label.json"
    test_img_dir: str = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/test_data_images"
    output_json: str = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/predictions.json"
    checkpoint_dir: str = "/home/vault/iwfa/iwfa110h/LAYOUT_LMV3_T5_SMALL"
    epoch: int = 30
    batch_size: int = 8
    chunk_size: int = 1000
    max_output_length: int = 512

class OcrInferenceDataset(Dataset):
    """Dataset for inference."""
    
    def __init__(self, data_list: list, image_dir: str):
        self.data_list = data_list
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        item = self.data_list[idx]
        img_name = item["img_name"]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        words = item["src_word_list"]
        boxes = item["src_wordbox_list"]
        return {
            "image": image,
            "words": words,
            "boxes": boxes,
            "img_name": img_name
        }

@dataclass
class InferenceCollator:
    """Collator for inference batches."""
    
    processor: object
    
    def __call__(self, features: list) -> dict:
        images = [f["image"] for f in features]
        words = [f["words"] for f in features]
        boxes = [f["boxes"] for f in features]
        img_names = [f["img_name"] for f in features]

        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return {
            "pixel_values": encoding["pixel_values"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "img_names": img_names
        }

def run_inference(config: InferenceConfig, device: Optional[torch.device] = None):
    """Run inference with given configuration."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()
    
    # Paths
    model_ckpt = os.path.join(config.checkpoint_dir, f"model_epoch_{config.epoch}.pth")
    proc_ckpt = os.path.join(config.checkpoint_dir, f"processor_epoch_{config.epoch}")
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(proc_ckpt, apply_ocr=False)
    t5_tokenizer = AutoTokenizer.from_pretrained(model_config.t5_model_name)
    
    # Initialize models
    layout_model = LayoutLMv3Model.from_pretrained(model_config.layoutlm_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_config.t5_model_name)
    
    # Build projection
    projection = torch.nn.Sequential(
        torch.nn.Linear(768, t5_model.config.d_model),
        LayerNorm(t5_model.config.d_model),
        GELU()
    )
    
    # Load checkpoint
    ckpt = torch.load(model_ckpt, map_location="cpu")
    layout_model.load_state_dict(ckpt['layout_model'])
    t5_model.load_state_dict(ckpt['t5_model'])
    projection.load_state_dict(ckpt['projection'])
    
    # Move to device and set eval mode
    layout_model.to(device).eval()
    t5_model.to(device).eval()
    projection.to(device).eval()
    
    # Prepare output
    writer = open(config.output_json, "w", encoding="utf-8")
    writer.write("{\n")
    first_entry = True
    
    # Process data
    for chunk in iter_ndjson_in_chunks(config.test_json, chunk_size=config.chunk_size):
        ds = OcrInferenceDataset(chunk, config.test_img_dir)
        loader = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=InferenceCollator(processor)
        )

        for batch in loader:
            pv = batch["pixel_values"].to(device)
            mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            img_names = batch["img_names"]
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                lm_out = layout_model(
                    pixel_values=pv,
                    input_ids=input_ids,
                    attention_mask=mask,
                    bbox=bbox
                )
                seq_len = input_ids.size(1)
                text_feats = lm_out.last_hidden_state[:, :seq_len, :]
                proj_feats = projection(text_feats)

                gen_ids = t5_model.generate(
                    inputs_embeds=proj_feats,
                    attention_mask=mask,
                    max_length=config.max_output_length
                )

            texts = t5_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            # Write results
            for img_name, txt in zip(img_names, texts):
                if not first_entry:
                    writer.write(",\n")
                first_entry = False
                writer.write(f"{json.dumps(img_name)}: {json.dumps(txt, ensure_ascii=False)}")

            writer.flush()
            
            # Clean up
            del pv, mask, bbox, lm_out, text_feats, proj_feats, gen_ids, input_ids
            torch.cuda.empty_cache()

        del ds, loader
        torch.cuda.empty_cache()

    writer.write("\n}")
    writer.close()
    print(f"Inference complete — results written to {config.output_json}")

def start_execution():
    """Main function to run inference."""
    config = InferenceConfig()
    run_inference(config)