# segment_utils.py

import jieba

def jieba_segment(text: str) -> str:
    tokens = list(jieba.cut(text, cut_all=False))
    return " ".join(tokens)
