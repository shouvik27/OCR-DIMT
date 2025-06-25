import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

def load_cleaned_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_valid_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_bleu_scores(cleaned_texts, valid_data):
    scores = {}
    smoothie = SmoothingFunction().method4
    
    # Create a mapping from img_name to ordered_src_doc (converted to string)
    valid_mapping = {item['img_name']: ' '.join(item['ordered_src_doc']) for item in valid_data}
    
    for img_name, cleaned_text in cleaned_texts.items():
        if img_name in valid_mapping:
            # Tokenize both texts properly
            reference = valid_mapping[img_name]
            reference_tokens = [word_tokenize(reference.lower())]
            candidate_tokens = word_tokenize(cleaned_text.lower())
            
            # Calculate BLEU score
            score = sentence_bleu(
                reference_tokens, 
                candidate_tokens,
                smoothing_function=smoothie,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            scores[img_name] = score
    
    return scores

def main():
    nltk.download('punkt')  # Download tokenizer data if needed
    
    # Load the data
    cleaned_texts = load_cleaned_texts('extracted_texts_mistral_ocr_validationset.json')
    valid_data = load_valid_dataset(r"D:\\Data\\valid_dataset.json")
    
    # Calculate BLEU scores
    bleu_scores = calculate_bleu_scores(cleaned_texts, valid_data)
    
    # Print individual scores
    for img_name, score in bleu_scores.items():
        print(f"{img_name}: BLEU score = {score:.4f}")
    
    # Calculate and print average score
    if bleu_scores:
        avg_score = sum(bleu_scores.values()) / len(bleu_scores)
        print(f"\nAverage BLEU score: {avg_score:.4f}")
    else:
        print("No matching documents found for BLEU score calculation.")

if __name__ == "__main__":
    main()