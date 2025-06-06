# AdaDec
<!-- ASE 2025 Submission -->

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

### 3. Evaluate Model

#### Greedy Decoding:

```bash
python src/eval/evaluate.py --model <model_name> --dataset <dataset_name>
```

#### Beam Search (beam size = 3):

```bash
python src/eval/evaluate.py --model <model_name> --beam 3 --dataset <dataset_name>
```

#### AdaFixL Decoding (lookahead length = 5):

```bash
python src/eval/evaluate.py --model <model_name> --decoding_mode AdaFixL --dataset <dataset_name>
```

#### AdaDynL Decoding(max lookahead length = 10):

```bash
python src/eval/evaluate.py --model <model_name> --decoding_mode AdaDynL --lookahead_length 10 --dataset <dataset_name>
```

#### Example Commands

* Evaluate `deepseek-1.3b` on `humaneval` using AdaFixL:

  ```bash
  python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaFixL --dataset humaneval
  ```

* Evaluate `qwen3-1.7b` on `mbpp` using Beam Search:

  ```bash
  python src/eval/evaluate.py --model qwen3-1.7b --beam 3 --dataset mbpp
  ```

#### Arguments for `evaluate.py`

| Argument                | Description                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--model`               | Model name. Options: `deepseek-1.3b`, `deepseek-6.7b`, `stable-3b`, `codellama-7b`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`.                      |
| `--dataset`             | Dataset to evaluate on. Options: `humaneval`, `mbpp`.                                                                                                        |
| `--beam`                | Beam size. Use `1` for greedy decoding; values >1 for beam search. Default: `1`.                                                                             |
| `--decoding_mode`       | Decoding strategy. Options: `Traditional`, `AdaFixL`, `AdaDynL`.                                                                                             |
| `--entropy_threshold`   | Entropy threshold. Options: `'Learned'` or a numeric value (e.g., `0.8`). Default: `Learned`.                                                                |
| `--max_new_tokens`      | Maximum number of new tokens to generate. Default: `512`.                                                                                                    |
| `--lookahead_length`    | For `AdaFixL`: fixed lookahead length; for `AdaDynL`: maximum lookahead length. Default: `5`.                                                                |
| `--lookahead_beam_size` | Lookahead beam size. Default: `3`.                                                                                                                           |
| `--logging_detail`      | If enabled, logs detailed information for each decoding step (e.g., entropy, score, whether lookahead is used, the actual lookahead length). Note: log files may be large. |
| `--dirname`             | Custom sub-directory name under `experiments/<dataset>_outputs/<model_name>/` to store result files.  Default: `new`.                                                             |

## Result Archive

All original model outputs—including generated code, pass@1 scores, decoding traces, and latency metrics—are bundled in the `experiments.zip` archive.
