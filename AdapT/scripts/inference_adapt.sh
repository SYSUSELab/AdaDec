#!/bin/bash

# This script is used to test the inference with Hugging Face models.

GPU=$1
PROMPT_FILE=$2
STOP_WORDS_JSON=$3
OUTPUT_FILE=$4
TEMP=$5
t_1=$6
t_2=$7
MODEL_NAME=${8:-"deepseek-ai/deepseek-coder-1.3b-instruct"}  # Default model if not specified

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

# Validation of required parameters
if [ -z "$GPU" ]; then
  echo "Please specify GPU ID."
  exit 1
fi
if [ -z "$PROMPT_FILE" ]; then
  echo "Please specify prompt file."
  exit 1
fi
PROMPT_FILE=$(realpath "$PROMPT_FILE")
if [ -z "$OUTPUT_FILE" ]; then
  echo "Please specify output file."
  exit 1
fi
OUTPUT_FILE=$(realpath "$OUTPUT_FILE")
if [ -z "$TEMP" ]; then
  echo "Please specify temperature"
  exit 1
fi
if [ -z "$STOP_WORDS_JSON" ]; then
  echo "Please specify stop words json"
  exit 1
fi
STOP_WORDS_JSON=$(realpath "$STOP_WORDS_JSON")

# Set CUDA environment (adjust CUDA_HOME path as needed)
export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=$GPU

# Build command for HF model inference
CMD="python -u $MAIN_DIR/tests/inference_adapt.py \
        --prompt-file $PROMPT_FILE \
        --output-file $OUTPUT_FILE \
        --model-name $MODEL_NAME \
        --out-seq-length 512 \
        --temperature $TEMP \
        --t_1 $t_1 \
        --t_2 $t_2 \
        --top-p 0.95 \
        --top-k 0 \
        --sample-n 1 \
        --stop-words-json $STOP_WORDS_JSON"

echo "Running command:"
echo "$CMD"
echo "Using model: $MODEL_NAME"
echo "GPU: $GPU"
echo ""

eval "$CMD"