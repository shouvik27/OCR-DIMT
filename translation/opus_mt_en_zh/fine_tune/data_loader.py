import json
from sklearn.model_selection import train_test_split
from config import SEED

def load_json_or_jsonl(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
    except (json.JSONDecodeError, ValueError):
        examples = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    examples.append(obj)
                except json.JSONDecodeError:
                    buffer = line
                    while True:
                        next_part = f.readline()
                        if not next_part:
                            break
                        buffer += next_part
                        try:
                            obj = json.loads(buffer)
                            examples.append(obj)
                            break
                        except json.JSONDecodeError:
                            continue
        return examples

def load_and_split(json_path, test_size=0.1, seed=SEED):
    raw = load_json_or_jsonl(json_path)
    filtered = [
        e for e in raw
        if (
            isinstance(e, dict)
            and "ordered_src_doc" in e
            and "doc_translation" in e
            and e["ordered_src_doc"]
            and e["doc_translation"]
        )
    ]
    train_examples, val_examples = train_test_split(
        filtered, test_size=test_size, random_state=seed
    )
    return train_examples, val_examples
