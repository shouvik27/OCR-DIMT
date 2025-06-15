import os
from dataclasses import dataclass

@dataclass
class Config:
    # Default paths
    TRAIN_JSON = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/train_dataset.json"
    TRAIN_IMG_DIR = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/output_train_img"
    TEST_JSON = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/testset_wo_label.json"
    TEST_IMG_DIR = "/home/hpc/iwfa/iwfa110h/Uddipan/ocr/Data/Data/test_data_images"
    
    # Model paths
    CHECKPOINT_DIR = "/home/vault/iwfa/iwfa110h/LAYOUT_LMV3_T5_SMALL"
    
    # Hyperparameters
    NUM_EPOCHS = 100
    MAX_SAMPLES = 6000
    CHUNK_SIZE = 1000
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_PCT = 0.1
    MAX_OUTPUT_LENGTH = 512
    GRADIENT_CLIP = 1.0
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.CHECKPOINT_DIR, "logs"), exist_ok=True)