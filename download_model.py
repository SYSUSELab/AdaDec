import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import traceback

LOG_FILE = "download_errors.log"

def log_error(model_path, error):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"模型 {model_path} 下载失败：\n")
        f.write(error)
        f.write("\n" + "="*80 + "\n")

def download_model(model_path, use_llama=False):
    print(f"开始下载模型: {model_path}")
    try:
        if use_llama:
            _ = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            _ = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        _ = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"模型 {model_path} 下载完成。")
    except Exception as e:
        print(f"Error: 模型 {model_path} 下载失败。错误信息已写入 {LOG_FILE}")
        log_error(model_path, traceback.format_exc())

if __name__ == "__main__":
    models = {
        "codellama/CodeLlama-7b-Python-hf": {"use_llama": True},
        "deepseek-ai/deepseek-coder-1.3b-instruct": {"use_llama": False},
        "deepseek-ai/deepseek-coder-6.7b-instruct": {"use_llama": False},
        "stabilityai/stable-code-instruct-3b": {"use_llama": False},
        "Qwen/Qwen3-0.6B": {"use_llama": False},
        "Qwen/Qwen3-1.7B": {"use_llama": False},
        "Qwen/Qwen3-4B": {"use_llama": False},
        "Qwen/Qwen3-8B": {"use_llama": False},
    }

    for model_path, opts in models.items():
        download_model(model_path, use_llama=opts["use_llama"])
