# AdaDec

## Environment Setup

We recommend using conda to manage dependencies:

```bash
conda env create -f env.yml
conda activate adadec
````

## Usage

### 1. Generate GT-Guide Data

If you are testing a new model, you can generate GT-guide data (stored in `data/gt_guide_data`) using:

```bash
python src/learn_threshold/generate_data.py --model <model_name>
```

### 2. Learn Entropy Threshold via Logistic Regression

```bash
python src/learn_threshold/logistic_regression.py --model <model_name>
```

Learned thresholds are saved to:
`data/learned_thresholds.json`

### 3. Generation and Evaluation:

#### Greedy Decoding:

```bash
python src/eval/evaluate.py --model <model_name> --dataset <dataset_name>
```

#### AdaDec Decoding:

```bash
python src/eval/evaluate.py --model <model_name> --decoding_mode AdaFixL --dataset <dataset_name>
```

#### AdapT Decoding:

To run AdapT, please refer to the official implementation available at:  
[AdapT](https://github.com/LJ2lijia/AdapT)

After installing the AdapT repository, you can run evaluation with commands provided in their repo.

#### DevEval Evaluation:

To run evaluation on the DevEval benchmark, please use the official implementation provided by the DevEval authors:  
[DevEval](https://github.com/seketeam/DevEval)

#### Arguments for `evaluate.py`

| Argument                | Description                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--model`               | Model name. Options: `deepseek-1.3b`, `deepseek-6.7b`, `stable-3b`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`.                      |
| `--dataset`             | Dataset to evaluate on. Options: `humaneval+`, `mbpp+`, `deveval`.                                                                                                        |
| `--decoding_mode`       | Decoding strategy. Options: `Traditional`, `AdaFixL`.                                                                                             |
| `--entropy_threshold`   | Entropy threshold. Options: `'Learned'` or a numeric value (e.g., `1.2`). Default: `Learned`.                                                                |
| `--max_new_tokens`      | Maximum number of new tokens to generate. Default: `512`.                                                                                                    |
| `--lookahead_length`    | Maximum lookahead length(L). Default: `5`.                                                                |
| `--lookahead_beam_size` | Lookahead beam size(B). Default: `3`.                                                                                                                           |
| `--logging_detail`      | If enabled, logs detailed information for each decoding step (e.g., entropy, score, whether lookahead is used, the actual lookahead length). Note: log files may be large. |

## Result Archive

All original model outputs—including generated code, pass@1 scores, decoding traces, and latency metrics—are bundled in the `experiments.zip` archive.
