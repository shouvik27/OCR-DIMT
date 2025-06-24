"""Configuration file for LayoutLMv3 + T5 model training."""

import os
from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Training configuration dataclass."""
    train_json: str = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/train_dataset.json"
    train_img_dir: str = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/output_train_img"
    save_dir: str = "/home/vault/iwfa/iwfa110h/LAYOUT_LMV3_T5_SMALL"
    log_dir: str = os.path.join(save_dir, "logs")
    num_epochs: int = 100
    max_samples: int = 6000
    chunk_size: int = 1000
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    layoutlm_model_name: str = "microsoft/layoutlmv3-base"
    t5_model_name: str = "t5-small"