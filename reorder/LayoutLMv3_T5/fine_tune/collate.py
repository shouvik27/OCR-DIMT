"""Custom collator for LayoutLMv3 + T5 model."""

from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Dict, List

# Initialize tokenizer at module level
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

@dataclass
class CustomCollator:
    """Custom collator for processing batches of data."""
    
    processor: object  # LayoutLMv3 processor
    
    def __call__(self, features: List[Dict]) -> Dict:
        """Process a batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Dictionary containing processed batch data
        """
        images = [f["image"] for f in features]
        words = [f["words"] for f in features]
        boxes = [f["boxes"] for f in features]
        targets = [f["target"] for f in features]

        # Process with LayoutLMv3 processor
        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Tokenize targets with T5 tokenizer
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