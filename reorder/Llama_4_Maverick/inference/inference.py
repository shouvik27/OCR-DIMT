import json
import time
import dotenv
import os

from .image_utils import image_file_to_base64
from .ocr_client import get_groq_client, get_system_message
from .examples import get_few_shot_examples
from .process import perform_ocr_with_examples
from .config import DATA_DIR, JSON_PATH, OUTPUT_PATH

dotenv.load_dotenv()

def start_execution():
    client = get_groq_client()
    system_message = get_system_message()
    few_shot_examples = get_few_shot_examples()
    extracted_texts = {}

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue

            img_name = data.get("img_name")
            if not img_name:
                continue

            img_path = os.path.join(DATA_DIR, img_name)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            print(f"Processing {img_name}...")
            try:
                decoded_output = perform_ocr_with_examples(
                    img_path, client, system_message, few_shot_examples, image_file_to_base64
                )
                extracted_texts[img_name] = decoded_output
                print(f"Extracted text for {img_name}:\n{decoded_output}\n{'-'*40}")
                time.sleep(1)
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as json_file:
        json.dump(extracted_texts, json_file, indent=4)
    print(f"All extracted texts have been saved to {OUTPUT_PATH}.")
