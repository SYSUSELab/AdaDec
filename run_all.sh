#!/bin/bash
echo "开始运行所有任务..."

# 支持命令行参数
if [ $# -eq 2 ]; then
  USE_GPUS=$1
  IFS=' ' read -ra GPU_IDS <<< "$2"
else
  USE_GPUS=2
  GPU_IDS=(0 1)
fi

# 参数检查
TOTAL_GPUS=$(nvidia-smi -L | wc -l)
if [ $USE_GPUS -gt $TOTAL_GPUS ]; then
  echo "错误：要使用的GPU数量($USE_GPUS)超过了可用GPU数量($TOTAL_GPUS)"
  exit 1
fi

if [ ${#GPU_IDS[@]} -ne $USE_GPUS ]; then
  echo "错误：GPU_IDS数组长度(${#GPU_IDS[@]})与USE_GPUS($USE_GPUS)不匹配"
  exit 1
fi

echo "将使用 $USE_GPUS 个GPU: ${GPU_IDS[*]}"

########################################
# 评估任务配置（根目录脚本）
########################################
EVAL_MODELS=("qwen3-1.7b" "deepseek-6.7b" "qwen3-4b" "qwen3-8b")
EVAL_MODES=("Traditional" "AdaFixL")

########################################
# 推理任务配置（AdapT 脚本）
########################################
PROMPT_FILE="inputs/deveval_filtered_data.jsonl"
STOP_WORDS_JSON="inputs/deveval_stop_words.json"
TEMP=0.5
t_1=0.2
t_2=0.01

OUTPUT_DIR="deveval_outputs"
mkdir -p "$OUTPUT_DIR"

INFER_MODELS=(
    "deepseek-ai/deepseek-coder-1.3b-instruct"
    "stabilityai/stable-code-instruct-3b"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
)

########################################
# 通用任务执行函数
########################################
run_task() {
  local task_type=$1   # eval / infer
  local arg1=$2        # 模型名 或 模型路径
  local arg2=$3        # eval 模式 或 GPU ID
  local gpu_id=$4      # GPU ID

  if [ "$task_type" == "eval" ]; then
    local model=$arg1
    local mode=$arg2
    echo "[$(date '+%H:%M:%S')] GPU $gpu_id: 评估 $model - $mode 开始"
    CUDA_VISIBLE_DEVICES=$gpu_id python src/eval/evaluate.py \
      --model $model --decoding_mode $mode --dataset deveval
    echo "[$(date '+%H:%M:%S')] GPU $gpu_id: 评估 $model - $mode 完成"

  elif [ "$task_type" == "infer" ]; then
    local model_name=$arg1
    local clean_name=$(echo "$model_name" | sed 's/\//-/g')
    local output_file="$OUTPUT_DIR/${clean_name}.jsonl"
    echo "[$(date '+%H:%M:%S')] GPU $gpu_id: 推理 $model_name 开始"
    CUDA_VISIBLE_DEVICES=$gpu_id sh AdapT/scripts/inference_adapt.sh \
      $gpu_id $PROMPT_FILE $STOP_WORDS_JSON $output_file $TEMP $t_1 $t_2 $model_name
    echo "[$(date '+%H:%M:%S')] GPU $gpu_id: 推理 $model_name 完成 -> $output_file"
  fi
}

export -f run_task

########################################
# 生成统一任务列表
########################################
> task_list.txt

# 评估任务
for Model in "${EVAL_MODELS[@]}"; do
  for Mode in "${EVAL_MODES[@]}"; do
    for GPU in "${GPU_IDS[@]}"; do
      echo "eval $Model $Mode $GPU" >> task_list.txt
    done
  done
done

# 推理任务
for Model in "${INFER_MODELS[@]}"; do
  for GPU in "${GPU_IDS[@]}"; do
    echo "infer $Model X $GPU" >> task_list.txt
  done
done

########################################
# 并行执行
########################################
echo "开始并行执行任务..."
cat task_list.txt | parallel -j $USE_GPUS --lb run_task {1} {2} {3} {4}

rm task_list.txt
echo "所有任务已完成。"
