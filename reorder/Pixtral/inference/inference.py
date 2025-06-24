import os
import json
import time
from tqdm import tqdm

from .config import MODEL_ID, TEST_IMAGES_DIR, TESTSET_JSON_PATH, OUTPUT_JSON_PATH
from ..fine_tune import *
from .examples import get_few_shot_examples
from .process import perform_ocr_with_examples

def start_execution():
    client = get_mistral_client()
    few_shot_examples = get_few_shot_examples()
    extracted_texts = {}

    with open(TESTSET_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    for data in tqdm(test_data, desc="Processing images"):
        img_name = data.get("img_name")
        if not img_name:
            continue
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        print(f"Processing {img_name}...")
        try:
            decoded_output = perform_ocr_with_examples(
                client, img_path, few_shot_examples, MODEL_ID, image_file_to_base64
            )
            extracted_texts[img_name] = decoded_output
            print(f"Extracted text for {img_name}:\n{decoded_output}\n{'-'*40}")
            time.sleep(1)
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(extracted_texts, json_file, indent=4)
    print(f"All extracted texts have been saved to {OUTPUT_JSON_PATH}.")

