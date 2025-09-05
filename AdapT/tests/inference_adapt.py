import argparse
import torch
import re
from pathlib import Path
from json import loads, dumps
from tqdm import tqdm
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from codegeex.torch.inference import get_token_stream


def add_code_generation_args(parser):
    """Modified argument parser - removed CodeGeeX specific model architecture args"""
    group = parser.add_argument_group(title="code generation")
    
    # Remove CodeGeeX model architecture parameters, keep generation parameters
    group.add_argument(
        "--t_1",
        type=float,
        default=0.8,
    )
    group.add_argument(
        "--t_2",
        type=float,
        default=0.2,
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--sample-n",
        type=int,
        default=1,
        help="Number of samples to generate per prompt.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--prompt-file",
        type=str,
        required=True,
    )
    group.add_argument(
        "--output-file",
        type=str,
        required=True,
    )
    group.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        help="Hugging Face model name to use"
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )
    group.add_argument(
        '--stop-words',
        type=str,
        default="\nclass, \ndef, \n#",
        help='Stop words to stop the generation.'
    )
    group.add_argument(
        '--stop-words-json',
        type=Path,
        default='',
        help='Stop words to stop the generation.'
    )
    return parser


def truncate(completion):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n\n\n\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def extract_last_function_body(code_str: str) -> str:
    """
    从 code_str 中定位 '# The code to be completed is:' 之后的第一个 def/async def，
    提取该函数体（不含三引号文档字符串，不含函数外的代码）。
    返回提取出的文本（保留原始缩进），若找不到则返回空字符串。
    """
    lines = code_str.splitlines()
    # 找到标记行
    marker_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if '# The code to be completed is:' in ln:
            marker_idx = i
            break
    if marker_idx is None:
        return ""

    # 在标记之后找到第一个 def 或 async def
    def_idx: Optional[int] = None
    def_pattern = re.compile(r'^\s*(async\s+def|def)\s')
    for i in range(marker_idx + 1, len(lines)):
        if def_pattern.match(lines[i]):
            def_idx = i
            break
    if def_idx is None:
        return ""

    # 计算 def 行的缩进长度（把 tab 视为 4 空格）
    leading_ws = re.match(r'^(\s*)', lines[def_idx]).group(1)
    def_indent = len(leading_ws.expandtabs(4))

    # 找到函数头结束行（考虑括号匹配）
    paren_balance = 0
    header_end = None
    for j in range(def_idx, len(lines)):
        ln = lines[j]
        # 简单地统计括号（假设函数签名内不会有未闭合字符串等极端情况）
        paren_balance += ln.count('(') - ln.count(')')
        # 当括号平衡并且该行以冒号结尾时，函数头结束
        if paren_balance <= 0 and ln.rstrip().endswith(':'):
            header_end = j
            break
    if header_end is None:
        # 没找到冒号结尾，视作无效
        return ""

    # 函数体从 header_end + 1 开始
    body_start = header_end + 1
    # 跳过可能的空行，找到第一个非空行（如果没有非空行，则函数体为空）
    first_body_line = None
    for k in range(body_start, len(lines)):
        if lines[k].strip() != "":
            first_body_line = k
            break
    if first_body_line is None:
        return ""

    # 若第一个非空行的缩进不大于 def 的缩进，说明函数没有体
    first_body_indent = len(re.match(r'^(\s*)', lines[first_body_line]).group(1).expandtabs(4))
    if first_body_indent <= def_indent:
        return ""

    # 检查是否为三引号文档字符串起始
    triple_re = re.compile(r'^\s*(?:[rubfRUBF]{0,3})("""|\'\'\')')
    m = triple_re.match(lines[first_body_line])
    content_start = first_body_line
    if m:
        delim = m.group(1)
        # 如果同一行包含结束分隔符（出现两次），则文档串在同一行结束
        line_text = lines[first_body_line]
        # 找起始 delim 的位置（考虑前缀），再搜索是否在该行还有第二个 delim（结束）
        # 简化策略：如果当前行中 delim 出现次数 >= 2，则认为结束在同一行
        if line_text.count(delim) >= 2:
            content_start = first_body_line + 1
        else:
            # 向下寻找结束 delim
            end_idx = None
            for t in range(first_body_line + 1, len(lines)):
                if delim in lines[t]:
                    end_idx = t
                    break
            if end_idx is None:
                # 文档串未闭合——把从 end 到 EOF 都视作文档串（返回空或剩余代码视情况）
                return ""
            content_start = end_idx + 1

        # 找到 content_start 后，可能全是空行，继续下面逻辑

    # 从 content_start 开始收集属于函数体的行
    collected = []
    for idx in range(content_start, len(lines)):
        ln = lines[idx]
        # 空行总是可以作为函数体的一部分
        if ln.strip() == "":
            collected.append(ln)
            continue
        # 计算此行缩进
        indent = len(re.match(r'^(\s*)', ln).group(1).expandtabs(4))
        # 如果缩进小于或等于 def 的缩进，说明函数体结束（遇到下一块代码）
        if indent <= def_indent:
            break
        collected.append(ln)

    # 去掉开头和结尾多余的空行（不改变内部相对缩进）
    # 保持至少一行（如果 collected 全为空行，则返回空字符串）
    while collected and collected[0].strip() == "":
        collected.pop(0)
    while collected and collected[-1].strip() == "":
        collected.pop(-1)

    return "\n".join(collected)


def sample_sequence_batch(
    model,
    tokenizer,
    context_tokens,
    context_lengths,
    attention_mask,
    position_ids,
    seq_length,
    out_seq_length,
    temp,
    return_scores=False,
    prompt_length=None,
    bad_ids=None,
    temperature=1.0,
    topp=1.0,
    topk=0,
    greedy=False,
    recompute=False,
):
    """Sample sequence batch - adapted for HF models"""
    batch_size = context_tokens.size(0)
    is_done = torch.zeros([batch_size]).byte().cuda()
    tokens = context_tokens.clone()
    
    with torch.no_grad():
        context_length = context_lengths.min().item()
        
        while context_length < (out_seq_length + prompt_length if prompt_length else seq_length):
            if context_length >= tokens.size(1):
                break
            
            # Prepare inputs for HF model
            current_tokens = tokens[:, :context_length]
            
            # For HF models, we don't pass position_ids unless the model specifically needs them
            # Most HF models handle position encoding internally
            model_inputs = {
                "input_ids": current_tokens,
                "use_cache": False
            }
            
            # Only add attention_mask if it's needed and properly formatted
            if attention_mask is not None:
                # Convert attention mask format for HF models (should be 1 for attend, 0 for not attend)
                hf_attention_mask = torch.ones_like(current_tokens)
                # Set padding positions to 0
                hf_attention_mask[current_tokens == tokenizer.eos_token_id] = 0
                model_inputs["attention_mask"] = hf_attention_mask
                
            # Forward pass through model
            outputs = model(**model_inputs)
            logits = outputs.logits
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if topk > 0:
                indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
                
            if topp < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > topp
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if greedy:
                next_tokens = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update tokens
            if context_length < tokens.size(1):
                tokens[:, context_length] = next_tokens
            else:
                tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for EOS
            is_done = is_done | (next_tokens == tokenizer.eos_token_id).byte()
            
            yield tokens, context_lengths
            
            if is_done.all():
                break
                
            context_length += 1


def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()

    temperature = args.temperature
    greedy = temperature == 0.0

    if greedy:
        temperature = 0.0

    temp = [args.t_1, args.t_2]

    # Load tokenizer - replace CodeGeeXTokenizer with HF AutoTokenizer
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.stop_words_json.exists():
        STOP_WORDS = loads(args.stop_words_json.read_text())
    else:
        STOP_WORDS = [word.strip() for word in args.stop_words.split(",")]

    def get_stop_regex(stop_words):
        return re.compile(rf"(.*?)(?:{'|'.join(re.escape(word) for word in stop_words)}).*", re.DOTALL)

    STOP_REGEX = get_stop_regex(STOP_WORDS)

    # Load model - replace CodeGeeX model loading with HF model loading
    print(f"Loading model {args.model_name} ...")
    if "CodeLlama" in args.model_name:
        model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    model.eval()

    # Optional quantization (you might need to install bitsandbytes)
    if args.quantize:
        print("Note: Quantization with HF models requires bitsandbytes. Skipping quantization.")

    with open(args.prompt_file, "r") as f:
        prompts = [loads(line) for line in f.readlines()]

    out_seq_length = args.out_seq_length
    seq_length = args.max_position_embeddings
    
    # Keep original generation logic
    bar = tqdm(total=len(prompts) * args.sample_n)
    with open(args.output_file, "w") as f:
        for json in prompts:
            json['completion'] = []
            prompt = json['prompt']
            if 'example' in json:
                prompt = json['example'] + '\n' + prompt
            
            if 'Qwen3' in args.model_name:
                prompt = '/no_think\n' + prompt
            
            # Replace tokenizer.encode_code with standard encode
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            n_token_prompt = len(tokens)
            
            for _ in range(args.sample_n):
                token_stream = get_token_stream(
                    model,
                    tokenizer,
                    seq_length,
                    out_seq_length,
                    [copy.deepcopy(tokens)],
                    temp=temp,
                    micro_batch_size=1,
                    topk=args.top_k,
                    topp=args.top_p,
                    temperature=temperature,
                    greedy=greedy,
                )
                for generated_tokens, _ in token_stream:
                    if generated_tokens[0].cpu().numpy()[-1] == tokenizer.eos_token_id:
                        break
                    elif len(generated_tokens[0]) >= out_seq_length + n_token_prompt:
                        break
                    elif (
                        len(generated_tokens[0]) >= n_token_prompt + 2
                        and STOP_REGEX.match("".join(tokenizer.decode(generated_tokens[0].cpu().numpy().tolist()[n_token_prompt:], skip_special_tokens=True)))
                    ):
                        break
                    else:
                        pass
                # else:
                    # raise RuntimeError("Failed to generate code.")
                generated_tokens_ = generated_tokens[0].cpu().numpy().tolist()

                if generated_tokens_[-1] == tokenizer.eos_token_id:
                    generated_tokens_ = generated_tokens_[:-1]
                
                # Replace tokenizer.decode_code with standard decode
                generated_code = tokenizer.decode(generated_tokens_[n_token_prompt:], skip_special_tokens=True)
                
                
                if STOP_REGEX.match(generated_code) and "deveval" not in args.prompt_file:
                    generated_code = STOP_REGEX.match(generated_code).group(1)
                
                if "deveval" in args.prompt_file:
                    f.write(dumps(dict(namespace=json['namespace'], completion=extract_last_function_body(prompt + '\n' + generated_code))) + '\n')
                else:
                    f.write(dumps(dict(task_id=json['task_id'], prompt=prompt, completion=truncate(generated_code), language='python')) + '\n')
                bar.update(1)

    bar.close()


if __name__ == "__main__":
    main()