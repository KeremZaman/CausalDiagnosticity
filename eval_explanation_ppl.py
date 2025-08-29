from eval import Evaluator
import json
import argparse
from evaluate import load
import os
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import logging


# Adapted from https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
def compute_ppl(predictions, model_id, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None):

    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None: # and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def _prepare_edited_prompt(edits, expl, tokenizer):
    edit_txt = ""
    for edit in edits:
        edit_txt += f"New Fact: {edit}\n"
    edit_prompt = f"Please acknowledge the following new facts and use them to answer the question:\n{edit_txt}"
    messages = [{"role": "user", "content": edit_prompt}, {"role": "assistant", "content": expl}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, continue_final_message=True, tokenize=False)
    return prompt

TASKS = [('fact_check', 1000), ('object_counting', 1000), ('analogy', 1000), ('multihop', 200)]

def calculate_ppl(model_name, edit_method, output):
    results = {}
    for task, num_samples in TASKS:
        if edit_method == 'memit':
            if task == 'multihop':
                continue
            full_model_name = f"{model_name}_edited_{task}_v1"
        else:
            full_model_name = model_name
        evaluator = Evaluator(task, full_model_name, model_name, num_samples)
        evaluator.prepare_data()
        synth_faithful_explanations, synth_unfaithful_explanations = evaluator.synthetic_explanations
        if edit_method == 'ice':
            edits_for_faithful_model, edits_for_unfaithful_model = evaluator._process_edits()
            edited_synth_faithful_explanations, edited_synth_unfaithful_explanations = [], []
            for expl, edit in zip(synth_faithful_explanations, edits_for_faithful_model):
                expl_with_edit = _prepare_edited_prompt(edit, expl, evaluator.tokenizer)
                edited_synth_faithful_explanations.append(expl_with_edit)
            
            for expl, edit in zip(synth_unfaithful_explanations, edits_for_faithful_model):
                expl_with_edit = _prepare_edited_prompt(edit, expl, evaluator.tokenizer)
                edited_synth_unfaithful_explanations.append(expl_with_edit)
            
            synth_faithful_explanations, synth_unfaithful_explanations = edited_synth_faithful_explanations, edited_synth_unfaithful_explanations

        faithful_expl_ppls = compute_ppl(model_id=full_model_name, predictions=synth_faithful_explanations, batch_size=1, add_start_token=False)
        unfaithful_expl_ppls = compute_ppl(model_id=full_model_name, predictions=synth_unfaithful_explanations, batch_size=1, add_start_token=False)
        avg_faithful_expl_ppl = faithful_expl_ppls['mean_perplexity']
        avg_unfaithful_expl_ppl = unfaithful_expl_ppls['mean_perplexity']
        lower_ppl_freq = np.mean(np.array(faithful_expl_ppls['perplexities']) < np.array(unfaithful_expl_ppls['perplexities']))
        results[task] = {'faithful_ppl': avg_faithful_expl_ppl, 'unfaithful_ppl': avg_unfaithful_expl_ppl, 
                         'faithful_expl_ppls': np.array(faithful_expl_ppls['perplexities']).tolist(),
                         'unfaithful_expl_ppls': np.array(unfaithful_expl_ppls['perplexities']).tolist(),
                         'lower_ppl_freq': lower_ppl_freq}

    with open(output, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--edit-method', required=True)
    parser.add_argument('--results-output', required=True)

    args = parser.parse_args()
    calculate_ppl(args.model_name, args.edit_method, args.results_output)
