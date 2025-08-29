# A Causal Lens for Evaluating Faithfulness Metrics

[![Arxiv](https://img.shields.io/badge/arXiv-2502.18848-b31b1b.svg)](https://arxiv.org/abs/2502.18848)
[![Hugging Face](https://img.shields.io/badge/🤗-Models%20%26%20Dataset-yellow)](https://huggingface.co/collections/l3-unc/causal-diagnosticity-68a4f6c59cb7a96f6398b2ba)
[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Authors: [Kerem Zaman](https://keremzaman.com) and [Shashank Srivastava](https://ssriva.com)  

---

## Overview

Large Language Models (LLMs) offer natural language explanations as an alternative to feature attribution methods for model interpretability. However, despite their plausibility, they may not reflect the model's true reasoning faithfully, which is crucial for understanding the model's true decision-making processes. Although several faithfulness metrics have been proposed, they are often evaluated in isolation, making direct, principled comparisons between them difficult. Here, we present Causal Diagnosticity, a framework \begin{updated} that serves as a common testbed \end{updated} to evaluate faithfulness metrics for natural language explanations. Our framework employs the concept of diagnosticity, and uses model-editing methods to generate faithful-unfaithful explanation pairs. Our benchmark includes four tasks: fact-checking, analogy, object counting, and multi-hop reasoning. We evaluate prominent faithfulness metrics, including post-hoc explanation and chain-of-thought-based methods. We find that diagnostic performance varies across tasks and models, with Filler Tokens performing best overall. Additionally, continuous metrics are generally more diagnostic than binary ones but can be sensitive to noise and model choice. Our results highlight the need for more robust faithfulness metrics. 


## Updates

- **Aug 20, 2025** Our [paper](https://arxiv.org/abs/2502.18848) has been accepted to EMNLP 2025!


## Leaderboard

### Main Results (Synthetic explanations + ICE)

#### Qwen2.5-7B

| Metric            | FactCheck | Analogy | Object Counting | Multi-hop |
|-------------------|-----------|---------|-----------------|-----------|
| **posthoc**          |           |         |                 |           |
| CC-SHAP           | **0.554** | 0.345   | **0.551**       | 0.438     |
| Simulatability    | 0.501     | **0.501** | 0.499           | **0.502** |
| **CoT**           |           |         |                 |           |
| Early Answering   | 0.756     | 0.534   | 0.566           | 0.468     |
| &nbsp; Early Answering (modified)   | 0.937     | 0.387   | 0.588           | 0.657     |
| Filler Tokens     | 0.828| 0.561   | 0.630       | 0.682 |
| &nbsp; Filler Tokens (NR, `$`)       | **0.948** | **0.786**   | **0.669**       | **0.748** |
| Adding Mistakes   | 0.534     | 0.590| 0.614           | 0.542     |
| Paraphrasing      | 0.556     | 0.535   | 0.425           | 0.448     |
| CC-SHAP           | 0.559     | 0.318   | 0.539           | 0.442     |

#### Gemma-2-9B-it

| Metric            | FactCheck | Analogy | Object Counting | Multi-hop |
|-------------------|-----------|---------|-----------------|-----------|
| **p.h.**          |           |         |                 |           |
| CC-SHAP           | **0.540** | **0.898** | 0.466           | **0.658** |
| Simulatability    | 0.507     | 0.501   | **0.500**       | 0.512     |
| **CoT**           |           |         |                 |           |
| Early Answering   | 0.838     | 0.859   | 0.724           | 0.435     |
| &nbsp; Early Answering (modified)  | 0.916     | 0.924   | 0.670           | 0.667     |
| Filler Tokens     | 0.893 | 0.810   | 0.843       | 0.585 |
| Filler Tokens (NR, `$`) | **0.936** | **0.962** | **0.855** | **0.778** |
| Adding Mistakes   | 0.427     | 0.639   | 0.579           | 0.402     |
| Paraphrasing      | 0.525     | 0.430   | 0.385           | 0.525     |
| CC-SHAP           | 0.598     | 0.939 | 0.506           | 0.488     |

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KeremZaman/CausalDiagnosticity.git
cd CausalDiagnosticity
pip install -r EasyEdit/requirements.txt
pip install -r requirements.txt
```

## Usage

### Running Experiments

#### Main Experiments (Synthetic explanations + ICE)

```
bash scripts/run.sh {TASK_NAME} {NUM_SAMPLES} {MODEL_NAME} ice synthetic
```

- `TASK_NAME`: one of  `fact_check`, `analogy`, `object_counting`, `multihop`

- For full evaluation use `NUM_SAMPLES=1000` for `fact_check`, `analogy`, `object_counting` and `NUM_SAMPLES=200` for `multihop`.


**Example:**

```
bash scripts/run.sh fact_check 1000 Qwen/Qwen2.5-7B ice synthetic
```

#### MEMIT Ablation

Use `memit` instead of `ice`. Provide the common model prefix (before the version suffix). For example, for edited models based on qwen2.5-7b, run:

```
bash scripts/run.sh fact_check 1000 l3-unc/qwen2.5-7b memit synthetic
```

#### Binary Metric Ablation

Use `run_binary.sh` instead of `run.sh` to evaluate the **binary variants** of CoT-corruption–based metrics.
```
bash scripts/run_binary.sh {TASK_NAME} {NUM_SAMPLES} {MODEL_NAME} ice synthetic
```

#### Metric Sensitivity

Use `run_analysis.sh` instead of `run.sh` to evaluate different variants of Filler Tokens and Early Answering metrics.

```
bash scripts/run_binary.sh {TASK_NAME} {NUM_SAMPLES} {MODEL_NAME} ice synthetic
```

#### Model Generated Explanation Ablation

Use `real` instead of `synthetic`.

```
bash scripts/run.sh {TASK_NAME} {NUM_SAMPLES} {MODEL_NAME} ice real
```

#### More Customization

```
usage: eval.py [-h] --editor-type {memit,ice} --model MODEL --unfaithful-model UNFAITHFUL_MODEL --tokenizer TOKENIZER --task
               {fact_check,object_counting,analogy,multihop} --metric
               {cc_shap-posthoc,cc_shap-cot,simulatability,lanham-truncated,lanham-truncated-continuous,lanham-truncated-continuous-informed,lanham-mistakes,lanham-mistakes-continuous,lanham-paraphrase,lanham-paraphrase-continuous,lanham-filler,lanham-filler-continuous,lanham-filler-continuous-nr,lanham-filler-continuous-star,lanham-filler-continuous-star-nr,lanham-filler-continuous-dash,lanham-filler-continuous-dash-nr,lanham-filler-continuous-dollar,lanham-filler-continuous-dollar-nr,lanham-filler-continuous-pilcrow,lanham-filler-continuous-pilcrow-nr}
               [--num-samples NUM_SAMPLES] --results-output RESULTS_OUTPUT [--seed SEED]
               [--use-synthetic-explanations]
```

### Edit Reliability

You can evaluate edit reliability by comparing the perplexity of paired explanations.

```
python eval_explanation_ppl.py --model-name {MODEL_NAME} \
                               --edit-method {EDIT_METHOD} \
                               --results-output results/result.json
```

- `EDIT_METHOD`: choose from `memit` or `ice`.

- For MEMIT-edited models: provide the base model prefix (without version suffix).

```
python eval_explanation_ppl.py --model-name l3-unc/qwen2.5-7b \
                               --edit-method memit 
                               --results-output results/expl_ppl_results_qwen-2.5-7b_memit.json 
```

This will measure edit reliability for edited model pairs `qwen2.5-7b_edited_{TASK}_v1` and `qwen2.5-7b_edited_{TASK}_v2` across all available tasks.


## Datasets

Our datasets underwent substantial human revision, but the following scripts were used to generate their initial versions:

**FactCheck**

```
python dataset_generation/generate_fact_check.py \
  --editing-data datasets/fact_check/edits.json \
  --eval-data datasets/fact_check/fact_check_eval.json \
  --size 1000
```

**Analogy**

```
python dataset_generation/generate_analogy.py \
  --editing-data datasets/analogy_dataset/edits.json \
  --eval-data datasets/analogy_dataset/analogy_eval.json \
  --size 1000
```

**Object Counting**

```
python dataset_generation/generate_object_counting.py \
  --proto-relation-dataset datasets/object_counting/proto_relation.json \
  --relation-dataset datasets/object_counting/relations.json \
  --editing-data datasets/object_counting/edits.json \
  --dataset datasets/object_counting/object_counting_dataset.json \
  --min-samples-per-relation 100 \
  --max-samples-per-relation 100
```

**Multihop Reasoning**

```
python dataset_generation/generate_multihop.py \
  --counterfactuals-dataset datasets/multihop/multihop_counterfactuals.json \
  --num-samples 1000 \
  --generate counterfactual


python dataset_generation/generate_multihop.py \
  --counterfactuals-dataset datasets/multihop/multihop_counterfactuals_fixed_200.json \
  --final-dataset datasets/multihop/multihop_full_200.json  \
  --generate dataset
```


## API

See `eval.py` for the main project structure. The core component is the `Evaluator` object, which returns faithfulness scores, explanations, correctness, and predictions for each instance. The `eval_diagnosticity()` method compares two Evaluator objects—one using faithful explanations and the other using unfaithful ones—to compute diagnosticity scores.


## Adding Your Own Metric

Implement a new class in `metrics.py` inheriting from the `Metric` abstract class and add its shorthand name to the `TEST_TO_METRIC` dictionary in `metrics.py`.

## Citation

If you use this work, please cite:

```
@article{Zaman2025ACL,
  title={A Causal Lens for Evaluating Faithfulness Metrics},
  author={Kerem Zaman and Shashank Srivastava},
  journal={ArXiv},
  year={2025},
  volume={abs/2502.18848},
  url={https://api.semanticscholar.org/CorpusID:276618030}
}
```

## Contributing

We welcome contributions! Please open an issue or submit a pull request for:

- New metrics

- Bug fixes or improvements

## Acknowledgements

Our research builds upon several fantastic open-source projects. We'd like to extend our sincere gratitude to their creators and maintainers.

* The model editing component using **MEMIT** is adapted from the powerful [EasyEdit](https://github.com/zjunlp/EasyEdit) library.

* Our implementations of **CC-SHAP** and other faithfulness metrics are based on the official [CC-SHAP repository](https://github.com/Heidelberg-NLP/CC-SHAP), which we have significantly modified and extended for this work.