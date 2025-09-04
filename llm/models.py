import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import AutoTokenizer


def init_codellama7b(model_path="codellama/CodeLlama-7b-Python-hf", device="cuda"):
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = "left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_deepseek1b(model_path="deepseek-ai/deepseek-coder-1.3b-instruct", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_deepseek7b(model_path="deepseek-ai/deepseek-coder-6.7b-instruct", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_stable3b(model_path="stabilityai/stable-code-instruct-3b", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_600m(model_path="Qwen/Qwen3-0.6B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_2b(model_path="Qwen/Qwen3-1.7B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_4b(model_path="Qwen/Qwen3-4B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwen3_8b(model_path="Qwen/Qwen3-8B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

MODEL_FACTORY = {
    "codellama-7b": init_codellama7b,
    "deepseek-1.3b": init_deepseek1b,
    "deepseek-6.7b": init_deepseek7b,
    "stable-3b": init_stable3b,
    "qwen3-0.6b": init_qwen3_600m,
    "qwen3-1.7b": init_qwen3_2b,
    "qwen3-4b": init_qwen3_4b,
    "qwen3-8b": init_qwen3_8b,
}