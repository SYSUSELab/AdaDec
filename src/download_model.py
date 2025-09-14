import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback

LOG_FILE = "download_errors.log"

def log_error(model_path, error):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Model {model_path} download failed:\n")
        f.write(error)
        f.write("\n" + "="*80 + "\n")

def download_model(model_path):
    print(f"Starting to download model: {model_path}")
    try:
        _ = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        _ = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Model {model_path} downloaded successfully.")
    except Exception as e:
        print(f"Error: Failed to download {model_path}. Error logged to {LOG_FILE}")
        log_error(model_path, traceback.format_exc())

if __name__ == "__main__":
    models = [
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "stabilityai/stable-code-instruct-3b",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B"
    ]

    for model_path in models:
        download_model(model_path)
