
#!/bin/bash

# 使用方法: ./run_eval.sh GPU_ID
# 例如: ./run_eval.sh 0

if [ $# -eq 0 ]; then
    echo "用法: $0 GPU_ID"
    echo "例如: $0 0"
    exit 1
fi

GPU_ID=$1

# 设置GPU并运行
export CUDA_VISIBLE_DEVICES=$GPU_ID

python src/eval/evaluate_cot.py \
    --model deepseek-1.3b \
    --decoding_mode AdaDynL \
    --lookahead_length 50 \
    --dataset humaneval+ \
    --entropy_threshold 0.25 \
    --logging_detail \
    --lookahead_beam_size 5