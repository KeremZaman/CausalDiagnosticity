import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'EasyEdit'))
from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams, FTHyperParams
import os
import torch
import json
import argparse
from datasets import load_dataset

HYPERPARAMS_MAP = {
    'memit': MEMITHyperParams,
    'ft': FTHyperParams
}

def edit_model(editing_method, prompts, subjects, new_targets, hparams_file, old_targets = None):
    HyperParams = HYPERPARAMS_MAP.get(editing_method, None)
    if HyperParams is None:
        raise NotImplementedError(f"Editing method {editing_method} not supported.")
        
    hparams = HyperParams.from_hparams(hparams_file)
    editor = BaseEditor.from_hparams(hparams)

    if old_targets is None:
        metrics, edited_model, _ = editor.batch_edit(prompts=prompts, target_new=new_targets, subject=subjects, keep_original_weight=False)
    else:
        metrics, edited_model, _ = editor.batch_edit(prompts=prompts, ground_truth=old_targets, target_new=new_targets, subject=subjects, keep_original_weight=False)

    return edited_model

def run(editing_method: str, task: str, hparams_file: str, output_dir: str):
    data = load_dataset('l3-unc/CausalDiagnosticity', task, split='test')['related_edits']
    
    prompts, subjects, targets_1, targets_2, true_target = list(map(list, zip(*[item[0].values() for item in data])))
    
    print("EDIT WITH TARGETS-1...")
    edited_model_1 = edit_model(editing_method, prompts, subjects, targets_1, hparams_file)
    edited_model_1.save_pretrained(f'{output_dir}_v1', from_pt=True)

    del edited_model_1
    torch.cuda.empty_cache()
    
    print("EDIT WITH TARGETS-2...")
    edited_model_2 = edit_model(editing_method, prompts, subjects, targets_2, hparams_file)
    edited_model_2.save_pretrained(f'{output_dir}_v2', from_pt=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset for model editing and factual evaluation')
    parser.add_argument('--task', required=True)
    parser.add_argument('--editing-method', choices=list(HYPERPARAMS_MAP.keys()))
    parser.add_argument('--hparams', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    run(args.editing_method, args.task, args.hparams, args.output)