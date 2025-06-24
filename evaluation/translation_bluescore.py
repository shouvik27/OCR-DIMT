def compute_bleu_score(reference, candidate):
    """
    Compute the BLEU score for a given reference and candidate translation.
    
    Args:
        reference (list of str): The reference translation(s).
        candidate (str): The candidate translation to evaluate.
    
    Returns:
        float: The computed BLEU score.
    """
    from nltk.translate.bleu_score import sentence_bleu

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score


def evaluate_translations(reference_translations, candidate_translations):
    """
    Evaluate a list of candidate translations against reference translations.
    
    Args:
        reference_translations (list of list of str): A list of reference translations.
        candidate_translations (list of str): A list of candidate translations to evaluate.
    
    Returns:
        list of float: A list of BLEU scores for each candidate translation.
    """
    scores = []
    for reference, candidate in zip(reference_translations, candidate_translations):
        score = compute_bleu_score(reference, candidate)
        scores.append(score)
    return scores