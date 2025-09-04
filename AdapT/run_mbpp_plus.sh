#!/bin/bash

# Define all the variables upfront
GPU=2
PROMPT_FILE="inputs/mbpp_test.jsonl"
STOP_WORDS_JSON="inputs/mbpp_stop_words.json"
TEMP=0.5
t_1=0.2
t_2=0.01

# Create the output directories if they don't exist
OUTPUT_DIR="mbpp+_outputs"
RESULT_DIR="mbpp+_results"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULT_DIR"

# List of models to iterate through
MODELS=(
    "deepseek-ai/deepseek-coder-1.3b-instruct"
    "stabilityai/stable-code-instruct-3b"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
    "codellama/CodeLlama-7b-Python-hf"
)

# Loop through each model
for MODEL_NAME in "${MODELS[@]}"; do
    echo "---"
    echo "Starting inference for model: $MODEL_NAME"

    # Replace slashes with hyphens for a valid filename
    CLEAN_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/\//-/g')
    
    # Define the output file path for this specific model
    OUTPUT_FILE="$OUTPUT_DIR/${CLEAN_MODEL_NAME}.jsonl"
    RESULT_FILE="$RESULT_DIR/${CLEAN_MODEL_NAME}_result.jsonl"

    # Run the main script with the current model
    sh scripts/inference_adapt.sh "$GPU" "$PROMPT_FILE" "$STOP_WORDS_JSON" "$OUTPUT_FILE" "$TEMP" "$t_1" "$t_2" "$MODEL_NAME"

    echo "Finished inference for $MODEL_NAME. Output saved to: $OUTPUT_FILE"

    # Run evaluation on the generated completion file
    echo "Evaluating results for model: $MODEL_NAME"
    python evaluate_mbpp+.py --completion_file "$OUTPUT_FILE" --result_file "$RESULT_FILE"
    echo "Evaluation finished. Result saved to: $RESULT_FILE"
    echo "---"
done

echo "All models have been processed and evaluated."
