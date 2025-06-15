from dataclasses import dataclass
from transformers import AutoProcessor, AutoTokenizer

t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

@dataclass
class CustomCollator:
    """Collator for training batches."""
    processor: AutoProcessor

    def __call__(self, features):
        images = [f["image"] for f in features]
        words = [f["words"] for f in features]
        boxes = [f["boxes"] for f in features]
        targets = [f["target"] for f in features]

        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        labels = t5_tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids

        return {
            "pixel_values": encoding["pixel_values"],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "labels": labels
        }

@dataclass
class InferenceCollator:
    """Collator for inference batches."""
    processor: AutoProcessor

    def __call__(self, features):
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