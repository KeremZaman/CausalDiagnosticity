from SPARQLWrapper import SPARQLWrapper, JSON
from datasets import load_dataset
from transformers import pipeline, Conversation, set_seed
from functools import cache
import json
import os
import re
import argparse
from tqdm import tqdm
import random
import time

def get_alternatives_from_wiki_data(default_item, num_alternatives, max_samples=100):
    query = """SELECT ?item ?itemLabel WHERE {{
        ?item wdt:P31 ?superClass.
        wd:{item_id} wdt:P31 ?superClass.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        FILTER(?item != wd:{item_id}) }} LIMIT {num_samples}"""
    
    sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparqlwd.setQuery(query.format(item_id=default_item, num_samples=max_samples))
    sparqlwd.setReturnFormat(JSON)

    # if any network issue occured, wait and repeat until getting results
    while True:
        try:
            results = sparqlwd.query().convert()['results']['bindings']
        except:
            print(f"An error occured while fetching data. Trying again..")
            time.sleep(5)
            continue
        break

    alternatives = [res['itemLabel']['value'] for res in results] 
    # filter non-named entities
    alternatives = list(filter(lambda x: re.match(r"^[A-Z][0-9]{4,}", x) is None, alternatives))

    return random.sample(alternatives, num_alternatives)


def generate_question_from_sent(sent):
    messages = [
        {
            "role": "user",
            "content": "Please create a yes-no question from the given sentence. Here are some examples:\n"
                        "Sentence: Joe Biden is the president of the United States. Question: Is Joe Biden the president of the United States?\n"
                        "Sentence: They play rock. Question: Do they play rock?\n"
                        "Sentence: Quesadilla from Mexico. Question: Is quesadilla from Mexico?\n"
                        "Do not mention your assumptions or assesment towards correctness of question. Do not output anything else! Stick with the format.\n"
                        f"Sentence: {sent} Question: "
        }]
    
    resp = pipe(messages, max_new_tokens=128)[0]['generated_text'][-1]['content'].split('\n')[0]
    question = re.sub(r"\(.+\)$", "", resp)
    return question

def generate_synthetic_explanation(sentence, true_target, new_target):
    return f"{sentence.replace(true_target, new_target)}, not {true_target}."

def generate_dataset(edit_out: str, eval_out: str, num_samples: int, seed: int = 123):
    dataset = load_dataset('NeelNanda/counterfact-tracing', split='train').shuffle(seed=seed).select(range(num_samples))

    editing_dataset, eval_dataset = [], []

    for item in tqdm(dataset):
        prompt, subject = item['prompt'].strip(), item['subject'].strip()
        true_target, true_target_id = item['target_true'].strip(), item['target_true_id']
        
        t1, t2 = get_alternatives_from_wiki_data(true_target_id, 2)
        edit_item = {'subject': subject, 'true_target': true_target, 'target_1': t1, 'target_2': t2, 'prompt': prompt}

        sent = f'{prompt} {true_target}'
        question = generate_question_from_sent(sent).strip()
        if question is None:
            continue
        
        synth_expl_1 = generate_synthetic_explanation(sent, true_target, t1)
        synth_expl_2 = generate_synthetic_explanation(sent, true_target, t2)
        eval_item = {'subject': subject, 'true_target': true_target, 'target_1': t1, 'target_2': t2, 
                     'question': question, 'synthetic_explanation_1': synth_expl_1, 'synthetic_explanation_2': synth_expl_2,
                     'related_edits': [edit_item]}
        
        editing_dataset.append(edit_item)
        eval_dataset.append(eval_item)

    with open(edit_out, 'w') as f:
        json.dump(editing_dataset, f, indent=4)
    
    with open(eval_out, 'w') as f:
        json.dump(eval_dataset, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset for model editing and factual evaluation')
    parser.add_argument('--editing-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=False, default=123)
    parser.add_argument('--model-name', required=False, default='Mistral-7B-Instruct-v0.2')

    args = parser.parse_args()

    set_seed(args.seed)
    pipe = pipeline('text-generation', model=os.path.join(os.environ['PRETRAINED_MODELS'], args.model_name), device=0)
    generate_dataset(args.editing_data, args.eval_data, args.size, args.seed)
