# translate_reordered.py

import torch
from config import (
    MODEL_DIR,
    REORDERED_VAL_JSON,
    REORDERED_TEST_JSON,
    OUT_TRANSLATED_VAL,
    OUT_TRANSLATED_TEST,
    BATCH_SIZE,
    MAX_LENGTH,
    NUM_BEAMS,
)
from io_utils import load_dict_json, save_dict_json
from segment_utils import jieba_segment
from translation_engine import load_translation_model, translate_dict

def start_execution():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading fine-tuned model from", MODEL_DIR)
    tokenizer, model = load_translation_model(MODEL_DIR, device)

    # Validation set
    val_dict = load_dict_json(REORDERED_VAL_JSON)
    print(f"Loaded {len(val_dict)} validation entries.")
    val_translated = translate_dict(
        val_dict, tokenizer, model, device, BATCH_SIZE, MAX_LENGTH, NUM_BEAMS, segment_func=jieba_segment, progress_desc="Validation"
    )
    save_dict_json(val_translated, OUT_TRANSLATED_VAL)
    print(f"Saved segmented translations to {OUT_TRANSLATED_VAL}")

    # Test set
    test_dict = load_dict_json(REORDERED_TEST_JSON)
    print(f"Loaded {len(test_dict)} test entries.")
    test_translated = translate_dict(
        test_dict, tokenizer, model, device, BATCH_SIZE, MAX_LENGTH, NUM_BEAMS, segment_func=jieba_segment, progress_desc="Test"
    )
    save_dict_json(test_translated, OUT_TRANSLATED_TEST)
    print(f"Saved segmented translations to {OUT_TRANSLATED_TEST}")

