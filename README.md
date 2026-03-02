# AdaDec

## Environment Setup

We recommend using conda to manage dependencies:

```bash
conda env create -f env.yml
conda activate adadec
```

## Quick Reproduction

To directly reproduce the pass@1 results from our paper without re-running the full pipeline, unpack the provided data and run the evaluation script:

```bash
unzip experiments.zip
python evaluate_pass1.py
```

## Full Pipeline Usage

To run the complete generation and evaluation process, follow the steps below.

### 1. Generate GT-Guide Data

If you are testing a new model, generate GT-guide data (stored in `data/gt_guide_data`) using:

```bash
python src/learn_threshold/generate_data.py --model <model_name>
```

### 2. Learn Entropy Threshold via Logistic Regression

```bash
python src/learn_threshold/logistic_regression.py --model <model_name>
```

Learned thresholds are saved to: `data/learned_thresholds.json`

### 3. Generation and Evaluation

#### Greedy:

```bash
python src/evaluate.py --model <model_name> --dataset <dataset_name>
```

#### Beam Search:

```bash
python src/eval/evaluate.py --model <model_name> --dataset <dataset_name> --beam 3
```

#### AdaDec:

```bash
python src/evaluate.py --model <model_name> --decoding_mode AdaFixL --dataset <dataset_name>
```

#### AdapT:

To run AdapT, please refer to their official implementation: [AdapT](https://github.com/LJ2lijia/AdapT). Run evaluation with the commands provided in their repo.

#### DevEval Evaluation:

To run evaluation on the DevEval benchmark, please use the official implementation provided by the authors: [DevEval](https://github.com/seketeam/DevEval).

#### Arguments for `evaluate.py`

| Argument | Description |
| --- | --- |
| `--model` | Model name. Options: `deepseek-1.3b`, `deepseek-6.7b`, `stable-3b`, `qwen2.5-1.5b`, `qwen2.5-7b`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`. |
| `--dataset` | Dataset to evaluate on. Options: `humaneval+` or `mbpp+`. |
| `--decoding_mode` | Decoding strategy. Options: `Traditional`, `AdaFixL`. |
| `--beam` | Beam size. Default: `1`. |
| `--entropy_threshold` | Entropy threshold. Options: `'Learned'` or a numeric value (e.g., `1.2`). Default: `Learned`. |
| `--max_new_tokens` | Maximum number of new tokens to generate. Default: `1024`. |
| `--lookahead_length` | Maximum lookahead length(L). Default: `5`. |
| `--lookahead_beam_size` | Lookahead beam size(B). Default: `3`. |
| `--logging_detail` | If enabled, logs detailed info for each decoding step (e.g., entropy, score, lookahead usage). Note: log files may be large. |

## Result Archive

All original model outputs and results are bundled in the `experiments.zip` archive.