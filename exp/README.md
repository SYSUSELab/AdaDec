# Experiments

This directory contains the raw experimental results for code generation tasks using various models on the **HumanEval** and **MBPP** benchmarks. For each model, we evaluate five decoding strategies and store the corresponding output files.

## Decoding Strategies

* **greedy**: Greedy decoding.
* **beamsearch**: Beam search with `beam_size=3`.
* **adaFixL**: AdaDec with a fixed lookahead length of 5.
* **adaDynL**: AdaDec with a dynamic lookahead length (up to 10).
* **fix\_threshold**: AdaDec with a fixed entropy threshold of 1.0.

## Structure

The folder structure is organized as follows:

```
experiments/
├── humaneval_outputs/
│   ├── <model_name>/
│   │   ├── <decoding_strategy>/
│   │   │   ├── <model_name>.jsonl
│   │   │   └── <model_name>.log
├── mbpp_outputs/
│   ├── <model_name>/
│   │   ├── <decoding_strategy>/
│   │   │   ├── <model_name>.jsonl
│   │   │   └── <model_name>.log
```

## Notes

Each `log` file contains:

  * Detailed decoding traces for each problem.
  * Final summary statistics, including:

    * `pass@1` results (number and percentage of problems passed).
    * List of failed problem IDs.
    * Total decoding time.
