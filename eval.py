from metrics import *
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset
from prompter import Prompter
import numpy as np
import argparse
from accelerate import Accelerator
import json
import os

torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

from transformers.utils import logging
logging.set_verbosity_error()

LABELS = {
    'fact_check': ['Yes', 'No'],
    'object_counting': ['A', 'B'],
    'analogy': ['A', 'B'],
    'multihop': ['A', 'B']
}

TASKS = list(LABELS.keys())

TEST_TO_METRIC = {
    'cc_shap-posthoc': CCShapPosthoc,
    'cc_shap-cot': CCShapCoT,
    'simulatability': Simulatability,
    'lanham-truncated': LanhamTruncated,
    'lanham-truncated-continuous': LanhamTruncatedContinuous,
    'lanham-truncated-continuous-informed': LanhamTruncatedContinuousInformed,
    'lanham-mistakes': LanhamMistakes,
    'lanham-mistakes-continuous': LanhamMistakesContinuous,
    'lanham-paraphrase': LanhamParaphrase,
    'lanham-paraphrase-continuous': LanhamParaphraseContinuous,
    'lanham-filler': LanhamFiller,
    'lanham-filler-continuous': LanhamFillerContinuous,
    'lanham-filler-continuous-nr': LanhamFillerContinuousNR,
    'lanham-filler-continuous-star': LanhamFillerContinuousStar,
    'lanham-filler-continuous-star-nr': LanhamFillerContinuousStarNR,
    'lanham-filler-continuous-dash': LanhamFillerContinuousDash,
    'lanham-filler-continuous-dash-nr': LanhamFillerContinuousDashNR,
    'lanham-filler-continuous-dollar': LanhamFillerContinuousDollarSign,
    'lanham-filler-continuous-dollar-nr': LanhamFillerContinuousDollarSignNR,
    'lanham-filler-continuous-pilcrow': LanhamFillerContinuousPilcrow,
    'lanham-filler-continuous-pilcrow-nr': LanhamFillerContinuousPilcrowNR,
}

class Evaluator(object):
    def __init__(self, task, model_name, tokenizer_name, num_samples):
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False, padding_side='left')
        self.prompter = Prompter(tokenizer_name, task)
        self.task = task
        self.num_samples = num_samples
        self.formatted_inputs, self.labels = [], []
        self.edits = []

    def load_model(self, max_new_tokens: int = 40):
        dtype = torch.float16
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map="auto")
        
        model.generation_config.is_decoder = True
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.min_new_tokens = 1
        # model.generation_config.do_sample = False
        model.config.is_decoder = True # for older models, such as gpt2
        model.config.max_new_tokens = max_new_tokens
        model.config.min_new_tokens = 1
        
        return model

    def _process_edits(self):
        if self.task not in TASKS:
            raise NotImplementedError
        dataset = load_dataset('l3-unc/CausalDiagnosticity', self.task, split='test').select(range(self.num_samples))
        edits = dataset['related_edits']

        faithful_edits_list, unfaithful_edits_list =  [], []
        for edit_items in edits:
            faithful_edits = []
            unfaithful_edits = []
            for item in edit_items:
                faithful_edit = item['edit_1'].strip() if self.task == 'multihop' else f"{item['prompt']} {item['target_1']}.".strip()
                unfaithful_edit = item['edit_2'].strip() if self.task == 'multihop' else f"{item['prompt']} {item['target_2']}.".strip()
                faithful_edits.append(faithful_edit)
                unfaithful_edits.append(unfaithful_edit)
            
            faithful_edits_list.append(faithful_edits)
            unfaithful_edits_list.append(unfaithful_edits)
    
        return faithful_edits_list, unfaithful_edits_list

    def prepare_data(self):
        print("Preparing data...")
        if self.task not in TASKS:
            raise NotImplementedError
        
        dataset = load_dataset('l3-unc/CausalDiagnosticity', self.task, split='test').select(range(self.num_samples))
        if self.task == 'analogy':
            dataset = dataset.map(lambda example: {"question": f"Fill in the blank: {example['question']} (A) {example['choice_A']} (B) {example['choice_B']}. Answer?" })
        elif self.task == 'object_counting':
            dataset = dataset.map(lambda example: {"question": f"{example['question']} (A) {example['choice_A']} (B) {example['choice_B']}. Answer?" })
        elif self.task == 'multihop':
            dataset = dataset.map(lambda example: {"question": f"{example['question']} (A) {example['choice_A']} (B) {example['choice_B']}. Answer?" })
        
        inputs, labels = dataset['question'], dataset['label']
        self.formatted_inputs, self.labels = inputs, labels
        self.synthetic_explanations = dataset['synthetic_explanation_1'], dataset['synthetic_explanation_2']

    def eval_faithfulness(self, metric_name, seed, pre_generated_explanations = None):
        set_seed(seed)
        self.model = self.load_model()
        self.prepare_data()
        metric = TEST_TO_METRIC[metric_name](self.model, self.tokenizer, self.prompter)
        faitfhulness_scores, generated_explanations = [], []
        total_score, count = 0, len(self.formatted_inputs)
        accuracy = 0.0
        predictions = []

        for i in tqdm(range(count)):
            formatted_input, label = self.formatted_inputs[i], self.labels[i]
            edits = self.edits[i] if len(self.edits) != 0 else []
            pre_generated_explanation = pre_generated_explanations[i] if pre_generated_explanations is not None else None
            # post-hoc tests
            score, generated_explanation, pred = metric.get_score(formatted_input, labels=LABELS[self.task], 
                                                                explanation=pre_generated_explanation, edits=edits)
            
            total_score += score
            predictions.append(pred)
            faitfhulness_scores.append(score)
            generated_explanations.append(generated_explanation)
            accuracy += 1.0 if label == pred else 0.0

        accuracy /= count
        print(f"Ran {metric_name} on {self.task} data.")
        print(f'Faithfulness score: {total_score/count:.2f}')
        print(f'Accuracy: {accuracy}')

        return faitfhulness_scores, accuracy, predictions, generated_explanations

def eval_diagnosticity(metric, evaluator: Evaluator, unfaithful_evaluator: Evaluator, results_file: str, seed: int, use_synthetic_explanations: bool = False):
    if use_synthetic_explanations:
        evaluator.prepare_data()
        synth_faithful_explanations, synth_unfaithful_explanations = evaluator.synthetic_explanations
        faithful_expls_scores, faithful_acc, faithful_preds, generated_explanations = evaluator.eval_faithfulness(metric, seed, synth_faithful_explanations)
        unfaithful_expls_scores, unfaithful_acc, unfaithful_preds, generated_unfaithful_explanations = evaluator.eval_faithfulness(metric, seed, synth_unfaithful_explanations)
    else:
        _, unfaithful_acc, _, generated_unfaithful_explanations = unfaithful_evaluator.eval_faithfulness(metric, seed)
        faithful_expls_scores, faithful_acc, faithful_preds, generated_explanations = evaluator.eval_faithfulness(metric, seed)
        unfaithful_expls_scores, unfaithful_acc, unfaithful_preds, generated_unfaithful_explanations = evaluator.eval_faithfulness(metric, seed, generated_unfaithful_explanations)

    faithful_expls_scores, unfaithful_expls_scores = np.array(faithful_expls_scores), np.array(unfaithful_expls_scores)
    #diagnosticity = np.sum(faithful_expls_scores > unfaithful_expls_scores) / faithful_expls_scores.size
    diagnosticity = np.mean((faithful_expls_scores > unfaithful_expls_scores).astype(int) + 0.5 * (faithful_expls_scores == unfaithful_expls_scores).astype(int))

    print(f"Generated explanations: {generated_explanations}")
    print(f"faithful scores: {faithful_expls_scores}")
    print(f"Generated unfaithful explanations: {generated_unfaithful_explanations}")
    print(f"unfaithful sscores: {unfaithful_expls_scores}")
    print(f"Diagnosticity: {diagnosticity}")

    results = {'metric': metric, 'diagnosticity': diagnosticity, 
                'faithful_accuracy': faithful_acc, 'unfaithful_acc': unfaithful_acc,
                'faithful_preds': faithful_preds, 'unfaithful_preds': unfaithful_preds,
               'faithful_explanations': generated_explanations, 
               'faithfulness_scores_for_faithful_explanations': faithful_expls_scores.tolist(),
               'unfaithful_explanations': generated_unfaithful_explanations,
               'faithfulness_scores_for_unfaithful_explanations': unfaithful_expls_scores.tolist()
               }

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return diagnosticity

def run(task, editor_type, metric_name, model_name, unfaithful_model_name, tokenizer_name, num_samples, results_output, seed = 42, use_synthetic_explanations = False):
    evaluator = Evaluator(task, model_name, tokenizer_name, num_samples)
    unfaithful_evaluator = Evaluator(task, unfaithful_model_name, tokenizer_name, num_samples)
    if editor_type == 'ice':
        edits_for_faithful_model, edits_for_unfaithful_model = evaluator._process_edits()
        evaluator.edits = edits_for_faithful_model
        unfaithful_evaluator.edits = edits_for_unfaithful_model

    eval_diagnosticity(metric_name, evaluator, unfaithful_evaluator, results_output, seed, use_synthetic_explanations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--editor-type', choices=['memit', 'ice'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--unfaithful-model', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--metric', choices=TEST_TO_METRIC.keys(), required=True)
    parser.add_argument('--num-samples', type=int, required=False, default=100)
    parser.add_argument('--results-output', required=True)
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--use-synthetic-explanations', action='store_true')

    args = parser.parse_args()
    run(args.task, args.editor_type, args.metric, args.model, args.unfaithful_model, args.tokenizer, args.num_samples, args.results_output, args.seed, args.use_synthetic_explanations)