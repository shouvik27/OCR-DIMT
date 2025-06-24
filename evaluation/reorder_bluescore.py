def compute_bleu_score(reference, hypothesis):
    # Implementation of BLEU score calculation
    # This is a placeholder for the actual BLEU score computation logic
    pass

def evaluate_reorder(reference_list, hypothesis_list):
    # Evaluate the BLEU score for a list of references and hypotheses
    bleu_scores = []
    for reference, hypothesis in zip(reference_list, hypothesis_list):
        score = compute_bleu_score(reference, hypothesis)
        bleu_scores.append(score)
    return bleu_scores

if __name__ == "__main__":
    # Example usage
    references = [["the cat is on the mat"], ["there is a cat on the mat"]]
    hypotheses = ["the cat is on the mat", "the cat is on the mat"]
    scores = evaluate_reorder(references, hypotheses)
    print("BLEU scores:", scores)