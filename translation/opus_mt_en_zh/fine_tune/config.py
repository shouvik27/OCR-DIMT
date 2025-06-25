MODEL_PATH = (
    "/home/vault/iwi5/iwi5294h/models/opus-mt-en-zh/"
    "models--Helsinki-NLP--opus-mt-en-zh/"
    "snapshots/408d9bc410a388e1d9aef112a2daba955b945255"
)
DATA_JSON  = "/home/vault/iwi5/iwi5294h/data_ocr/Dataset/jsons/train_dataset.json"
OUTPUT_DIR = "./finetuned_opus_en_zh"
MAX_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 75
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
SAVE_STEPS = 1000
SEED = 42
