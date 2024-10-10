# Imports

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from multiple_choice import (
    get_answer_multiple_choice,
    compare_answers,
    dynamic_multiple_choice_base_prompt
)
from multiple_choice_w_subtype import (
    get_answer_multiple_choice_w_subtype,
    compare_answers,
    dynamic_multiple_choice_subtype_base_prompt
)

from multiple_choice_w_dstype import get_answer_multiple_choice_w_dstype, compare_answers

from qa import (
    get_answer_qa,
    calculate_bleu_score,
    calculate_levenshtein_score,
    calculate_rouge_score
)
from rag import get_answer_rag




# Utility functions:

def handle_qa_score(actual_answer, predicted_answer):
    if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
        return 0
    
    score = (
        # 0.25 * int(float(get_evaluation_score(question, actual_answer, predicted_answer, benchmark_type=benchmark_type, model_name=model_name)))) \
        calculate_bleu_score(actual_answer, predicted_answer) \
        + calculate_rouge_score(actual_answer, predicted_answer) \
        + calculate_levenshtein_score(actual_answer, predicted_answer)
    )
    return score

def handle_context_qa_score(actual_answer, predicted_answer):
    if predicted_answer.lower() == 'long answer' or 'error' in predicted_answer.lower():
        return 0

    score = (
        # 0.25 * int(float(get_evaluation_score_context(question, actual_answer, predicted_answer, benchmark_type=benchmark_type, model_name=model_name)))) \
        calculate_bleu_score(actual_answer, predicted_answer) \
        + calculate_rouge_score(actual_answer, predicted_answer) \
        + calculate_levenshtein_score(actual_answer, predicted_answer)
    )

    return score





def evaluate(model_name, num_fewshot, batch_size, device, limit):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    # Result definition:

    result = {
        "config": {
            # "model_name": model_name,
            # "num_fewshot": num_fewshot,
            # "batch_size": batch_size,
            # "device": device,
            # "limit": limit,
        },
        "results": {}
    }

    


    # Metadata:

    # v1 without dstype
    datasets = [
        {
            "task_type": "mmlu",
            "group": "banking",
            "data": load_dataset("LLM-Beetle/Banking-benchmark_mmlu_fqa_latest")["train"]
        },
        {
            "task_type": "mmlu",
            "group": "mmlu",
            "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
        },
        {
            "task_type": "mmlu",
            "group": "banking",
            "data": load_dataset("LLM-Beetle/banking_support")["train"]
        },
        {
            "task_type": "mmlu",
            "group": "arc",
            "data": load_dataset("LLM-Beetle/arc_translated")["train"]
        },
        {
            "task_type": "mmlu_context",
            "group": "context",
            "data": load_dataset("LLM-Beetle/anl_quad")["train"]
        },
        {
            "task_type": "qa",
            "group": "economics",
            "data": load_dataset("LLM-Beetle/LLM_generated_qa_latest")["train"]
        },
        {
            "task_type": "rag",
            "group": "context",
            "data": load_dataset("LLM-Beetle/Quad_benchmark_cqa_latest")["train"]
        },
    ]


    # v2 without dstype and with subtypeText
    datasets = [
        {
            "task_type": "mmlu",
            "group": "banking",
            "data": load_dataset("LLM-Beetle/Banking-benchmark_mmlu_fqa_latest")["train"],
            "subtext": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
        },
        {
            "task_type": "mmlu",
            "group": "mmlu",
            "data": load_dataset("LLM-Beetle/mmlu-aze")["train"],
            "subtext": ""
        },
        {
            "task_type": "mmlu",
            "group": "banking",
            "data": load_dataset("LLM-Beetle/banking_support")["train"],
            "subtext": "You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
        },
        {
            "task_type": "mmlu",
            "group": "arc",
            "data": load_dataset("LLM-Beetle/arc_translated")["train"],
            "subtext": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. Your task is selecting the most accurate option in Azerbaijani based on a given question. Choose the single binary text (True or False). Respond with only the binary text (True or False), without any additional text."
        },
        # {
        #     "task_type": "mmlu_context",
        #     "group": "context",
        #     "data": load_dataset("LLM-Beetle/anl_quad")["train"],
        #     "subtext": "You are an AI designed to answer questions in Azerbaijani based on given context in Azerbaijani. Your task is checking if the current text is from the context or not. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
        # },
        {
            "task_type": "qa",
            "group": "economics",
            "data": load_dataset("LLM-Beetle/LLM_generated_qa_latest")["train"],
            "subtext": ""
        },
        {
            "task_type": "rag",
            "group": "context",
            "data": load_dataset("LLM-Beetle/Quad_benchmark_cqa_latest")["train"],
            "subtext": ""
        },
    ]

    # v3 with dstype
    datasets = [
        {
            "task_type": "mmlu",
            "dstype": "mc",
            "group": "banking",
            "data": load_dataset("LLM-Beetle/Banking-benchmark_mmlu_fqa_latest")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "kmc",
            "group": "mmlu",
            "data": load_dataset("LLM-Beetle/mmlu-aze")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "tc",
            "group": "banking",
            "data": load_dataset("LLM-Beetle/banking_support")["train"]
        },
        {
            "task_type": "mmlu",
            "dstype": "arc",
            "group": "arc",
            "data": load_dataset("LLM-Beetle/arc_translated")["train"]
        },
        {
            "task_type": "mmlu_context",
            "dstype": "qmc",
            "group": "context",
            "data": load_dataset("LLM-Beetle/anl_quad")["train"]
        },
        {
            "task_type": "qa",
            "dstype": "qa",
            "group": "economics",
            "data": load_dataset("LLM-Beetle/LLM_generated_qa_latest")["train"]
        },
        {
            "task_type": "rag",
            "dstype": "cqa",
            "group": "context",
            "data": load_dataset("LLM-Beetle/Quad_benchmark_cqa_latest")["train"]
        },
    ]






    # main runner:
    for dataset in datasets:
        task_type = dataset['task_type']
        data = dataset['data']
        subtext = dataset.get('subtext', '')
        dstype = data['dstype']
        group = data['group']

        total_score = 0
        total_limit = min(limit, len(data))  # if needed
        
        dataset_score = 0  # Score for the current dataset
        correct_count = 0  # Count of correct answers for the dataset

        
        base_prompt_dynamic_mmlu = dynamic_multiple_choice_base_prompt(dataset=data, few_shot=5)  # for dynamic version 
        base_prompt_dynamic_subtype_mmlu = dynamic_multiple_choice_subtype_base_prompt(
            dataset=data, few_shot=5, subtype_text=subtext
        )  # for dynamic version w subtext


        for i in tqdm(range(total_limit), desc=f'Evaluating {task_type}'): 
            question = data['text'][i]
            options = data.get('options', [])[i] if task_type == "mmlu" else None
            context = data.get('context', [])[i] if task_type in ["rag", "mmlu_context"] else None
            correct_answer = data['answer'][i]


            # ANSWERS:
            if task_type in ["mmlu", "mmlu_context"]:
                # if i >= num_fewshot: # if base few-shot works   # comment it if need to run all dataset

                    # predicted_answer = get_answer_multiple_choice(
                    #     question, options, model, tokenizer, num_fewshot, base_prompt_dynamic_mmlu
                    # ) # dynamic version

                    # predicted_answer = get_answer_multiple_choice_w_subtype(question, options, model, tokenizer, num_fewshot, base_prompt_dynamic_subtype_mmlu)  # dynamic, incl subtype

                    predicted_answer = get_answer_multiple_choice_w_dstype(question, options, model, tokenizer, num_fewshot, dstype=dstype)  # static, with dstype


            elif task_type == "qa":
                context = data.get('context', [''])[i]
                predicted_answer = get_answer_qa(question, model, tokenizer)

            elif task_type == "rag":
                context = data.get('context', [''])[i]
                predicted_answer = get_answer_rag(question, context, model, tokenizer)

            else:
                raise ValueError("Invalid task type")



            # SCORES:  
            # Version 1 (base version)
        #     if task_type in ["mmlu", "mmlu_context"]:
        #         if compare_answers(correct_answer, predicted_answer):
        #             total_score += 1

        #     elif task_type == "qa":
        #         score = handle_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
        #         # score = handle_qa_score(question=question, actual_answer=correct_answer, predicted_answer=predicted_answer)
        #         total_score += score

        #     elif task_type == "rag":
        #         score = handle_context_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
        #         # score = handle_qa_score(question=question, actual_answer=correct_answer, predicted_answer=predicted_answer)
        #         total_score += score

        # result["results"][task_type] = {
        #     "metric_name": total_score / total_limit if total_limit > 0 else 0.0
        # }    

        # result.save_to_disk('path/to/save/') # save
        # result.to_parquet('path/to/save/dataset.parquet')


            # SCORES  
            # Version 2: (base + dataset and group score)
            if task_type in ["mmlu", "mmlu_context"]:
                if compare_answers(correct_answer, predicted_answer):
                    total_score += 1
                    correct_count += 1  # Count correct answers for the dataset

            elif task_type == "qa":
                score = handle_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
                total_score += score
                dataset_score += score  # Update dataset score

            elif task_type == "rag":
                score = handle_context_qa_score(actual_answer=correct_answer, predicted_answer=predicted_answer)
                total_score += score
                dataset_score += score  # Update dataset score

        # Average score for the dataset
        average_dataset_score = dataset_score / total_limit if total_limit > 0 else 0.0
        result["dataset_scores"][f"{group}_{task_type}"] = average_dataset_score

        # Accumulate score for the group
        if group not in result["group_scores"]:
            result["group_scores"][group] = {
                "total_score": 0,
                "total_count": 0
            }
        result["group_scores"][group]["total_score"] += correct_count
        result["group_scores"][group]["total_count"] += total_limit

    # Compute average scores for each group
    for group, scores in result["group_scores"].items():
        total_score = scores["total_score"]
        total_count = scores["total_count"]
        result["group_scores"][group] = total_score / total_count if total_count > 0 else 0.0


    return result







if __name__ == '__main__':
    result = evaluate('LLM-Beetle/ProdataAI_Llama-3.1-8bit-Instruct', 0, 0, 'cpu', 10)
    print(result)
