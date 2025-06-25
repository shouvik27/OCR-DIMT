# compute_bleu.py

import json
import jieba
from nltk.translate.bleu_score import corpus_bleu

# ─── 1) PATH CONFIGURATION ─────────────────────────────────────────────────────

# 1A) Gold validation JSON: a JSON array (or JSONL) of dicts. Each dict must have:
#     • "img_name"        : string
#     • "doc_translation" : list of Chinese tokens (subwords/characters)
GOLD_VAL_JSON = "/home/vault/iwi5/iwi5294h/data_ocr/Dataset/jsons/validset.json"

# 1B) Predictions from translate_reordered.py:
#     a dict { "image_name.png": "space-separated Chinese text", … }
PREDICTED_VAL_JSON = "./Output/Validation_Set/translated_reordered_Pixtral_fewshot_val.json"


# ─── 2) UTILITY TO LOAD JSON OR JSONL ────────────────────────────────────────────

def load_json_or_jsonl(path: str):
    """
    Try to load `path` as a single JSON array. If that fails (JSONDecodeError or not a list),
    fallback to JSONL: parse each non‐empty line as a JSON object.
    Returns a Python list of dicts.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError("Expected a JSON array at the top level.")
    except (json.JSONDecodeError, ValueError):
        objs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    objs.append(obj)
                except json.JSONDecodeError:
                    continue
        return objs


# ─── 3) MAIN BLEU COMPUTATION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # 3A) Load gold data (list of dicts)
    gold_data = load_json_or_jsonl(GOLD_VAL_JSON)
    if not isinstance(gold_data, list):
        raise ValueError(f"{GOLD_VAL_JSON} could not be loaded as a list of dicts.")

    # Build gold_map: img_name → [ [ reference_token_list ] ]
    gold_map = {}
    for entry in gold_data:
        if not isinstance(entry, dict):
            continue
        img = entry.get("img_name")
        raw_ref_tokens = entry.get("doc_translation")
        if not (isinstance(img, str) and isinstance(raw_ref_tokens, list)):
            continue

        # Join raw tokens into one Chinese string, then re-segment with jieba
        ref_text = "".join(raw_ref_tokens)
        ref_tokens = list(jieba.cut(ref_text, cut_all=False))

        gold_map[img] = [ref_tokens]  # wrap in a list-of-lists for corpus_bleu

    print(f"Built gold_map with {len(gold_map)} entries from {GOLD_VAL_JSON}")

    # 3B) Load predicted translations: dict { img_name → “space-separated Chinese” }
    with open(PREDICTED_VAL_JSON, "r", encoding="utf-8") as f:
        pred_dict = json.load(f)
    if not isinstance(pred_dict, dict):
        raise ValueError(f"{PREDICTED_VAL_JSON} must be a JSON dict mapping img_name → pred_text.")

    print(f"Loaded {len(pred_dict)} model translations from {PREDICTED_VAL_JSON}")

    # 3C) Align gold ↔ pred by img_name, build (references, hypotheses) lists
    all_references = []
    all_hypotheses = []

    for img, hyp_str in pred_dict.items():
        if img not in gold_map:
            # Skip if no gold reference for this image_name
            continue

        refs = gold_map[img]  # [[ref_tok1, ref_tok2, …]]
        # Hypothesis tokens = hyp_str.split() because hyp_str is already space-separated
        hyp_tokens = hyp_str.split()

        all_references.append(refs)
        all_hypotheses.append(hyp_tokens)

    print(f"Scoring BLEU on {len(all_hypotheses)} matching examples…")
    bleu_score = corpus_bleu(
        all_references,
        all_hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4
    )

    print(f"\nValidation BLEU = {bleu_score * 100:.2f}\n")
