import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

def download_model(model_path, use_llama=False):
    print(f"开始下载模型: {model_path}")
    if use_llama:
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"模型 {model_path} 下载完成。")

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
    print("所有模型下载完成。")
