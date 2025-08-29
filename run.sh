#!/bin/bash

Models=("deepseek-1.3b" "stable-3b" "qwen3-0.6b" "qwen3-1.7b" "deepseek-6.7b" "codellama-7b" "qwen3-4b" "qwen3-8b" )
Modes=("Traditional" "AdaFixL")   # "Traditional" "AdaFixL" "AdaDynL"

for Model in "${Models[@]}"; do
  for Mode in "${Modes[@]}"; do
    python src/eval/evaluate.py --model $Model --decoding_mode $Mode --dataset deveval
  done

  # 特殊情况：Traditional + beam=3
#   python src/eval/evaluate.py --model $Model --decoding_mode Traditional --dataset deveval --beam 3
done






# MbppPlus

# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model qwen3-4b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model qwen3-8b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model stable-3b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model stable-3b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model codellama-7b --decoding_mode Traditional --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model codellama-7b --decoding_mode AdaFixL --dataset mbpp+ --logging_detail


# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model stable-3b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3
# python src/eval/evaluate.py --model codellama-7b --decoding_mode Traditional --dataset mbpp+ --logging_detail --beam 3




# HumanEvalPlus

# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model qwen3-4b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model qwen3-8b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model stable-3b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model stable-3b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail

# python src/eval/evaluate.py --model codellama-7b --decoding_mode Traditional --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model codellama-7b --decoding_mode AdaFixL --dataset humaneval+ --logging_detail


# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model stable-3b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3
# python src/eval/evaluate.py --model codellama-7b --decoding_mode Traditional --dataset humaneval+ --logging_detail --beam 3



# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model stable-3b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail
# python src/eval/evaluate.py --model codellama-7b --decoding_mode AdaDynL --dataset mbpp+ --logging_detail

# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model stable-3b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail
# python src/eval/evaluate.py --model codellama-7b --decoding_mode AdaDynL --dataset humaneval+ --logging_detail





# DevEval
# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model deepseek-1.3b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model deepseek-6.7b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model qwen3-0.6b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model qwen3-1.7b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model qwen3-4b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model qwen3-4b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model qwen3-8b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model qwen3-8b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model stable-3b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model stable-3b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model stable-3b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model stable-3b --decoding_mode Traditional --dataset deveval --beam 3

# python src/eval/evaluate.py --model codellama-7b --decoding_mode Traditional --dataset deveval
# python src/eval/evaluate.py --model codellama-7b --decoding_mode AdaFixL --dataset deveval
# python src/eval/evaluate.py --model codellama-7b --decoding_mode AdaDynL --dataset deveval
# python src/eval/evaluate.py --model codellama-7b --decoding_mode Traditional --dataset deveval --beam 3
