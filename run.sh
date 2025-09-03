#!/bin/bash



# echo "开始运行评估任务..."
# Models=("deepseek-1.3b" "stable-3b" "qwen3-0.6b" "qwen3-1.7b" "deepseek-6.7b" "codellama-7b" "qwen3-4b" "qwen3-8b")
# Datasets=("humaneval+" "mbpp+")

# # echo "正在激活 adadec 环境..."
# # conda activate adadec

# # 支持命令行参数
# if [ $# -eq 2 ]; then
#   USE_GPUS=$1
#   IFS=' ' read -ra GPU_IDS <<< "$2"
# else
#   # 默认配置：使用前2个GPU
#   USE_GPUS=2
#   GPU_IDS=(0 1)
# fi

# # 参数检查
# TOTAL_GPUS=$(nvidia-smi -L | wc -l)
# if [ $USE_GPUS -gt $TOTAL_GPUS ]; then
#   echo "错误：要使用的GPU数量($USE_GPUS)超过了可用GPU数量($TOTAL_GPUS)"
#   exit 1
# fi

# if [ ${#GPU_IDS[@]} -ne $USE_GPUS ]; then
#   echo "错误：GPU_IDS数组长度(${#GPU_IDS[@]})与USE_GPUS($USE_GPUS)不匹配"
#   exit 1
# fi

# echo "将使用 $USE_GPUS 个GPU: ${GPU_IDS[*]}"

# # 简单的任务执行函数
# run_task() {
#   local model=$1
#   local dataset=$2
#   local job_slot=$3
  
#   # 将GPU_IDS导入到函数内部
#   IFS=' ' read -ra LOCAL_GPU_IDS <<< "${GPU_IDS_STR}"
  
#   # 轮询分配GPU
#   local gpu_index=$(( (job_slot - 1) % USE_GPUS ))
#   local gpu_id=${LOCAL_GPU_IDS[$gpu_index]}
  
#   echo "[$(date '+%H:%M:%S')] GPU $gpu_id: $model - $dataset 开始"
#   CUDA_VISIBLE_DEVICES=$gpu_id python src/eval/evaluate_cot.py \
#     --model $model \
#     --decoding_mode AdaDynL \
#     --lookahead_length 20 \
#     --dataset $dataset \
#     --entropy_threshold 0.12 \
#     --logging_detail
#   echo "[$(date '+%H:%M:%S')] GPU $gpu_id: $model - $dataset 完成"
# }

# export -f run_task
# export USE_GPUS
# export GPU_IDS_STR="${GPU_IDS[*]}"

# # 生成任务列表
# > task_list.txt
# for Model in "${Models[@]}"; do
#   for Dataset in "${Datasets[@]}"; do
#     echo "$Model $Dataset" >> task_list.txt
#   done
# done

# # 并行执行
# echo "开始并行执行任务..."
# parallel -j $USE_GPUS --colsep ' ' run_task {1} {2} {%} :::: task_list.txt

# # 清理
# rm task_list.txt
# echo "所有评估任务已完成。"






# if [ ! -f .env_initialized ]; then
#     echo "正在进行首次设置：创建并激活 Conda 环境..."
#     conda env create -f env.yml
#     conda activate adadec
#     # 创建一个标记文件，以便下次运行时跳过初始化
#     touch .env_initialized
#     echo "初始化完成。"
# else
#     echo "Conda 环境已初始化。正在跳过..."
# fi


echo "开始运行评估任务..."
Models=("deepseek-1.3b" "stable-3b" "qwen3-0.6b" "qwen3-1.7b" "deepseek-6.7b" "codellama-7b" "qwen3-4b" "qwen3-8b") # "deepseek-1.3b" "stable-3b" "qwen3-0.6b" "qwen3-1.7b" "deepseek-6.7b" "codellama-7b" "qwen3-4b" "qwen3-8b"
Modes=("Traditional" "AdaFixL"  "AdaDynL")  # "Traditional" "AdaFixL"  "AdaDynL"

echo "正在激活 adadec 环境..."
conda activate adadec

# 支持命令行参数
if [ $# -eq 2 ]; then
  USE_GPUS=$1
  IFS=' ' read -ra GPU_IDS <<< "$2"
else
  # 默认配置：使用前2个GPU
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

# 简单的任务执行函数
run_task() {
  local model=$1
  local mode=$2
  local job_slot=$3
  
  # 将GPU_IDS导入到函数内部
  IFS=' ' read -ra LOCAL_GPU_IDS <<< "${GPU_IDS_STR}"
  
  # 轮询分配GPU
  local gpu_index=$(( (job_slot - 1) % USE_GPUS ))
  local gpu_id=${LOCAL_GPU_IDS[$gpu_index]}
  
  echo "[$(date '+%H:%M:%S')] GPU $gpu_id: $model - $mode 开始"
  echo "正在评估模型：$model，模式：$mode"

  if [ "$mode" == "AdaDynL" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python src/eval/evaluate.py \
      --model $model --decoding_mode $mode --dataset deveval --lookahead_length 10
  else
    CUDA_VISIBLE_DEVICES=$gpu_id python src/eval/evaluate.py \
      --model $model --decoding_mode $mode --dataset deveval
  fi

  echo "[$(date '+%H:%M:%S')] GPU $gpu_id: $model - $mode 完成"
}

export -f run_task
export USE_GPUS
export GPU_IDS_STR="${GPU_IDS[*]}"

# 生成任务列表
> task_list.txt
for Model in "${Models[@]}"; do
  for Mode in "${Modes[@]}"; do
    echo "$Model $Mode" >> task_list.txt
  done
done

# 并行执行
echo "开始并行执行任务..."
parallel -j $USE_GPUS --colsep ' ' run_task {1} {2} {%} :::: task_list.txt

# 清理
rm task_list.txt
echo "所有评估任务已完成。"

















# # --- 初始化部分：仅在首次运行时执行 ---
# # 检查是否已激活环境或已创建。这里使用一个简单的文件标记来判断。
# # 如果文件 '.env_initialized' 不存在，则执行初始化命令。
# if [ ! -f .env_initialized ]; then
#     echo "正在进行首次设置：创建并激活 Conda 环境..."
#     conda env create -f env.yml
#     conda activate adadec
#     # 创建一个标记文件，以便下次运行时跳过初始化
#     touch .env_initialized
#     echo "初始化完成。"
# else
#     echo "Conda 环境已初始化。正在跳过..."
# fi

# # --- 核心运行部分：可重复执行 ---
# echo "开始运行评估任务..."
# Models=("deepseek-1.3b" "stable-3b" "qwen3-0.6b" "qwen3-1.7b" "deepseek-6.7b" "codellama-7b" "qwen3-4b" "qwen3-8b")
# Modes=("Traditional" "AdaFixL" "AdaDynL")  # "Traditional" "AdaFixL" "AdaDynL"

# # 激活 Conda 环境，以防脚本从新终端运行
# # 这行在每次运行脚本时都会执行，以确保环境是激活的。
# echo "正在激活 adadec 环境..."
# conda activate adadec

# for Model in "${Models[@]}"; do
#     for Mode in "${Modes[@]}"; do
#         echo "正在评估模型：$Model，模式：$Mode"
#         python src/eval/evaluate.py --model $Model --decoding_mode $Mode --dataset deveval
#     done

#     # 特殊情况：Traditional + beam=3
#     # echo "正在为模型 $Model 运行Beam Search (beam=3)"
#     # python src/eval/evaluate.py --model $Model --decoding_mode Traditional --dataset deveval --beam 3
# done

# echo "所有评估任务已完成。"






















# Models=("deepseek-1.3b" "stable-3b" "qwen3-0.6b" "qwen3-1.7b" "deepseek-6.7b" "codellama-7b" "qwen3-4b" "qwen3-8b" )
# Modes=("Traditional" "AdaFixL" "AdaDynL")   # "Traditional" "AdaFixL" "AdaDynL"

# for Model in "${Models[@]}"; do
#   for Mode in "${Modes[@]}"; do
#     python src/eval/evaluate.py --model $Model --decoding_mode $Mode --dataset deveval
#   done

#   # 特殊情况：Traditional + beam=3
# #   python src/eval/evaluate.py --model $Model --decoding_mode Traditional --dataset deveval --beam 3
# done






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
