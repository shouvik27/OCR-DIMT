import os
import json
from tqdm import tqdm
from .image_utils import image_file_to_base64

def iter_ndjson_in_chunks(json_path, chunk_size=1000):
    with open(json_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunk.append(obj)
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
        if chunk:
            yield chunk

def prepare_training_data(json_path, image_dir, output_file, max_samples=None):
    training_data = []
    processed_count = 0
    print("Preparing training data...")

    for chunk in iter_ndjson_in_chunks(json_path):
        if max_samples and processed_count >= max_samples:
            break
        for item in tqdm(chunk, desc=f"Processing chunk (Total: {processed_count})"):
            if max_samples and processed_count >= max_samples:
                break
            try:
                img_name = item.get("img_name")
                if not img_name:
                    continue
                img_path = os.path.join(image_dir, img_name)
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                ordered_text = " ".join(item.get("ordered_src_doc", item.get("src_word_list", [])))
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an OCR assistant. For every provided image, extract the text in the correct reading order. Return your answer as a plain text string."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_file_to_base64(img_path)}
                                },
                                {
                                    "type": "text",
                                    "text": "Please extract the ordered text from this image."
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": f"Ordered Text output: {ordered_text}"
                        }
                    ]
                }
                training_data.append(training_example)
                processed_count += 1
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

    print(f"Prepared {len(training_data)} training examples")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    return output_file, len(training_data)

def validate_training_data(file_path, sample_size=5):
    print(f"Validating training data format in {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                data = json.loads(line)
                print(f"Sample {i+1}:")
                print(f"  Messages: {len(data.get('messages', []))}")
                print(f"  First message role: {data['messages'][0]['role'] if data.get('messages') else 'N/A'}")
                print(f"  Has image: {'image_url' in str(data)}\n")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {i+1}: {e}")
