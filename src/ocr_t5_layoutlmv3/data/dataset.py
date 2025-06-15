import os
from PIL import Image
from torch.utils.data import Dataset

class OcrReorderDataset(Dataset):
    """Dataset for training LayoutLMv3 + T5 model."""
    def __init__(self, data_list, image_dir, processor):
        self.data_list = data_list
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = os.path.join(self.image_dir, item["img_name"])
        image = Image.open(image_path).convert("RGB")
        words = item["src_word_list"]
        boxes = item["src_wordbox_list"]
        target = " ".join(item.get("ordered_src_doc", words))
        return {"image": image, "words": words, "boxes": boxes, "target": target}

class OcrInferenceDataset(Dataset):
    """Dataset for inference with LayoutLMv3 + T5 model."""
    def __init__(self, data_list, image_dir):
        self.data_list = data_list
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
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