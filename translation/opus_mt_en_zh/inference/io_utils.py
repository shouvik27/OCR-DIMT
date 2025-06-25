# io_utils.py

import json

def load_dict_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must be a JSON dict, got {type(data)}.")
        return data

def save_dict_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
