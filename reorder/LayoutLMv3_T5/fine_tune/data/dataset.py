"""Dataset classes for LayoutLMv3 + T5 model."""

import os
from typing import Dict, List
from PIL import Image
from torch.utils.data import Dataset

class OcrReorderDataset(Dataset):
    """Dataset for OCR reordering task using LayoutLMv3 and T5."""
    
    def __init__(self, data_list: List[Dict], image_dir: str, processor):
        """Initialize dataset.
        
        Args:
            data_list: List of data items (dictionaries)
            image_dir: Directory containing images
            processor: LayoutLMv3 processor
        """
        self.data_list = data_list
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self) -> int:
        """Return number of items in dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Dictionary containing image, words, boxes, and target text
        """
        item = self.data_list[idx]
        image_path = os.path.join(self.image_dir, item["img_name"])
        image = Image.open(image_path).convert("RGB")
        words = item["src_word_list"]
        boxes = item["src_wordbox_list"]
        target = " ".join(item.get("ordered_src_doc", words))
        return {"image": image, "words": words, "boxes": boxes, "target": target}