import numpy as np
import json
import pandas as pd
import os
from datasets import Dataset
from torch.nn.functional import softmax
import torch
import logging
import sys

def lm_generate(input, model, tokenizer, max_new_tokens=100, padding=False, repeat_input=True, generate_random=False):
    """ Generate text from a huggingface language model (LM).
    Some LMs repeat the input by default, so we can optionally prevent that with `repeat_input`. """
    device = next(model.parameters()).device
    input_ids = tokenizer([input], return_tensors="pt", padding=padding, add_special_tokens=False).input_ids.to(device)
    if not generate_random:
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False) #, do_sample=False, min_new_tokens=1, max_new_tokens=max_new_tokens)
    else:
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.2) #, do_sample=False, min_new_tokens=1, max_new_tokens=max_new_tokens)
    # prevent the model from repeating the input
    if not repeat_input:
        generated_ids = generated_ids[:, input_ids.shape[1]:]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]


def lm_classify(inputt, model, tokenizer, padding=False, labels=['A', 'B']):
    """ Choose the token from a list of `labels` to which the LM asigns highest probability.
    https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15  """
    device = next(model.parameters()).device
    input_ids = tokenizer([inputt], padding=padding, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    generated_ids = model.generate(input_ids, do_sample=False, output_scores=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)

    # find out which ids the labels have
    label_scores = np.zeros(len(labels))

    for i, label in enumerate(labels):
        idx = 0 if any([True if x in model.name_or_path.lower() else False for x in ['gpt', 'llama', 'mistral', 'qwen', 'gemma']]) else 1
        label_id = tokenizer.encode(label, add_special_tokens=False)[idx] # TODO: check this for all new models: print(tokenizer.encode(label))
        label_scores[i] = generated_ids.scores[0][0, label_id]
    
    # choose as label the one wih the highest score
    prediction = labels[np.argmax(label_scores)]
    pred_scores = softmax(torch.tensor(label_scores)).tolist()
    return prediction, pred_scores

def setup_logger(logger_name):
    """Sets up a logger with a stream handler to stdout."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        logger.addHandler(sh)

    return logger