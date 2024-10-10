from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_answer_rag(question, context, model, tokenizer):
    prompt = (
        f"You are an answer generator AI in Azerbaijani. Your task is to generate concise answers based on the provided context and the given question.\n"
        f"Provide a clear and accurate answer in Azerbaijani, limited to 1-2 sentences and under 400 characters.\n\n"
        "### CONTEXT ###\n\n"
        '"""\n\n'
        f"{context}\n\n"
        '"""\n\n'
        "### END CONTEXT ###\n\n"
        f"Question: {question} Answer in Azerbaijani:\n"
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



