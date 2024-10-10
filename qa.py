from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein



def get_answer_qa(question, context, model, tokenizer):
    prompt = (
        f"You are an AI designed to generate concise answers in Azerbaijani based on the following questions.\n"
        f"Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
        f"Question in Azerbaijani:\n{question}\n\n"
        f"Context in Azerbaijani:\n{context}\n\n"
        f"Answer in Azerbaijani:"
    )

    encoding = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            max_length=100,
            pad_token_id=tokenizer.pad_token_id,
        )

    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return predicted_answer






def calculate_bleu_score(actual_answer: str, predicted_answer: str) -> float:
    """
    Calculate BLEU score for the given actual and predicted answers.
    """
    reference = actual_answer.split()
    candidate = predicted_answer.split()
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
    print('bleu:', 33 * bleu_score)
    return 33 * bleu_score

def calculate_rouge_score(actual_answer: str, predicted_answer: str) -> dict:
    """
    Calculate ROUGE score for the given actual and predicted answers.
    """
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    scores = scorer.score(actual_answer, predicted_answer)

    normalized_scores = {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

    print('rouge:', 33 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL']))
    return 33 * (0.33 * normalized_scores['rouge1'] + 0.33 * normalized_scores['rouge2'] + 0.33 * normalized_scores['rougeL'])


def calculate_levenshtein_score(actual_answer: str, predicted_answer: str) -> int:
    """
    Calculate Levenshtein distance between actual and predicted answers.
    """

    max_len = max(len(actual_answer), len(predicted_answer))

    if max_len == 0:  # both strings are empty
        return 1

    print('levenshtein:', 33 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / max_len)))
    return 33 * (1 - (Levenshtein.distance(actual_answer, predicted_answer) / max_len))



