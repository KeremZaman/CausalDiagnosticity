from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
import re
import functools
from typing import List
from datasets import load_dataset
import json
import random
import argparse

client = OpenAI()
env = Environment(loader=FileSystemLoader('dataset_generation/prompts'))


def retry(limit: int = 3):
    """
    A decorator that retries the decorated function until it succeeds or the retry limit is reached.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < limit:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}")
                    if attempts >= limit:
                        print("Max retries reached. Raising exception.")
                        #raise
                        return None
        return wrapper
    return decorator

@retry(limit=3)
def get_explanation(question: str, answer: str, facts: List[str], decomposition: List[str]):
    template = env.get_template('explanation_generation.txt')
    prompt = template.render(question=question, answer=answer, facts=facts, decomposition=decomposition)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=1.0)
    output =  response.choices[0].message.content
    pattern = re.compile(r"^explanation:\s(.*)$", re.MULTILINE)
    matches = pattern.findall(output)

    if len(matches) == 0:
        raise ValueError("No matches found.")
    
    return matches[0]

@retry(limit=3)
def get_counterfactuals(question: str, answer: str, facts: List[str], decomposition: List[str]):
    template = env.get_template('counterfactual_generation.txt')
    prompt = template.render(question=question, answer=answer, facts=facts, decomposition=decomposition)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gpt-4o",
                                              messages=messages, temperature=1.0, seed=0)
    output =  response.choices[0].message.content
    
    lines = output.splitlines()
    cf_list_1, cf_list_2 = [], []
    current_list = None

    for line in lines:
        if line.startswith("counterfactuals-1"):
            current_list = cf_list_1
        elif line.startswith("counterfactuals-2"):
            current_list = cf_list_2
        else:
            if current_list is not None:
                cf = line.strip().lstrip("-").strip()
                if cf != "":
                    current_list.append(cf)

    if len(cf_list_1) == 0 or len(cf_list_2) == 0 or len(cf_list_1) != len(cf_list_2):
        raise ValueError("Failed to generate counterfactuals properly.")
    
    return cf_list_1, cf_list_2


def generate_counterfactuals(out_file: str, num_samples: int = 1000):
    # get strategy qa dataset
    data = load_dataset("tasksource/strategy-qa")["train"].shuffle(seed=42)
    new_dataset, i, total = [], 0, 0
    while i < num_samples:
        item = data[i]
        question, facts, decomposition = item["question"], item["facts"], item["decomposition"]
        answer = "yes" if item["answer"] else "no"
        counterfactuals = get_counterfactuals(question, answer, facts, decomposition)
        
        if counterfactuals is None:
            i += 1
            continue
        
        cf_list_1, cf_list_2 = counterfactuals
        
        new_dataset.append({
            "question": question,
            "answer": answer,
            "facts": facts,
            "decomposition": decomposition,
            "counterfactuals_1": cf_list_1,
            "counterfactuals_2": cf_list_2
        })

        i += 1
        total += 1
    
    with open(out_file, 'w') as f:
        json.dump(new_dataset, f, indent=4)


def generate_final_dataset(dataset_file: str, out_file: str):
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    new_dataset = []
    
    for item in dataset:
        question, answer, cf_list_1, cf_list_2, decomposition = item["question"], item["answer"], item["counterfactuals_1"], item["counterfactuals_2"], item["decomposition"]
        explanation_1 = get_explanation(question, answer, cf_list_1, decomposition)
        explanation_2 = get_explanation(question, answer, cf_list_2, decomposition)
        choices = ["yes", "no"]
        random.shuffle(choices)
        
        related_edits = []
        for cf_1, cf_2 in zip(cf_list_1, cf_list_2):
            related_edits.append({"edit_1": cf_1, "edit_2": cf_2})
        
        new_dataset.append({"question": question, "choice_A": choices[0], "choice_B": choices[1], 
                            "label": "A" if choices.index(answer) == 0 else "B", "label_txt": answer, 
                            "synthetic_explanation_1": explanation_1, "synthetic_explanation_2": explanation_2,
                            "related_edits": related_edits, "decomposition": decomposition})

        with open(out_file, "w") as f:
            json.dump(new_dataset, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to handle counterfactual and dataset operations.")
    
    parser.add_argument("--counterfactuals-dataset", required=False)
    parser.add_argument("--final-dataset", required=False)
    parser.add_argument("--num-samples", required=False, type=int, default=1000)
    parser.add_argument("--generate", action="store", choices=["counterfactual", "dataset"], required=True)
    
    args = parser.parse_args()

    if args.generate == "counterfactual":
        generate_counterfactuals(args.counterfactuals_dataset, args.num_samples)        
    elif args.generate == "dataset":
        generate_final_dataset(args.counterfactuals_dataset, args.final_dataset)
