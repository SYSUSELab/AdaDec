import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import argparse
import logging
import re
import sys
from pathlib import Path
import time
import json
import multiprocessing
from datasets import load_dataset
from typing import Optional
import builtins
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from llm.models import MODEL_FACTORY
from llm.generator import Generator
from llm.generator_deveval import Generator_DevEval



def extract_python_code(code_fragment: str) -> str:
    stop_pattern = re.compile(r'(Human|Assistant|User|System)')
    match = stop_pattern.search(code_fragment)
    if match:
        code_fragment = code_fragment[:match.start()]

    lines = code_fragment.splitlines()
    import_lines = []
    func_lines = []
    collecting_func = False

    for line in lines:
        if not collecting_func and re.match(r'^(import\s|from\s+\w+)', line.strip()):
            import_lines.append(line)
            continue

        if not collecting_func and re.match(r'^def\s+\w+\s*\(', line):
            collecting_func = True
            func_lines.append(line)
            continue

        if collecting_func:
            if line.strip() == '' or re.match(r'^\s+', line):
                func_lines.append(line)
            else:
                break

    if func_lines:
        return '\n'.join(import_lines + [''] + func_lines).rstrip()
    else:
        return ''

def build_prompt(example: dict) -> str:
    code = example["code"].strip().splitlines()

    def_index = None
    for i, line in enumerate(code):
        if line.strip().startswith("def "):
            def_index = i
            break
    if def_index is None:
        raise ValueError(f"code not found: {example['code'][:80]}")

    prefix_code = "\n".join(code[:def_index])

    def_line = code[def_index]

    docstring = example["prompt"].strip()

    prompt_parts = []
    if prefix_code:
        prompt_parts.append(prefix_code)  # import
    prompt_parts.append(f"""{def_line}
    \"\"\" {docstring} \"\"\"
    # YOUR CODE HERE""")

    return "\n".join(prompt_parts) + "\n"

def _exec_with_env(code: str, env: dict, problem: dict):
    exec(code, env, env)

    for imp in problem.get("test_imports", []):
        exec(imp, env, env)

    for t in problem.get("test_list", []):
        exec(t, env, env)

    exec(problem["test"], env, env)

def evaluate_generation_mbppplus(problem: dict, generated_code: str, timeout: int = 5):
    evaluation = False 

    env = {"__builtins__": __builtins__}

    def runner(return_dict):
        try:
            _exec_with_env(generated_code, env, problem)
            return_dict["success"] = True
        except Exception:
            return_dict["success"] = False

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=runner, args=(return_dict,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        evaluation = False
    else:
        evaluation = return_dict.get("success", False)

    return evaluation

def exec_with_env_humanevalplus(code: str, env: dict, problem: dict):
    exec(code, env, env)
    
    exec(problem["test"], env, env)
    
    entry_point = problem["entry_point"]
    if entry_point in env:
        env["check"](env[entry_point])
    else:
        raise NameError(f"Function '{entry_point}' not found in generated code")

def evaluate_generation_humanevalplus(problem: dict, generated_code: str, timeout: int = 5):
    evaluation = False
    
    env = {"__builtins__": builtins}
    
    def runner(return_dict):
        try:
            exec_with_env_humanevalplus(generated_code, env, problem)
            return_dict["success"] = True
        except Exception as e:
            return_dict["success"] = False
            return_dict["error"] = str(e)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=runner, args=(return_dict,))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        evaluation = False
    else:
        evaluation = return_dict.get("success", False)
    
    return evaluation

def summarize_results(results):
    total_problems = len(results)
    passed_problems = sum(1 for res in results if res["evaluation"])
    pass_rate = (passed_problems / total_problems) * 100 if total_problems > 0 else 0
    failed_problem_ids = [res["problem_id"] for res in results if not res["evaluation"]]

    logging.info(f"Total number of problems: {total_problems}")
    logging.info(f"Number of problems passed: {passed_problems}")
    logging.info(f"Pass rate: {pass_rate:.2f}%")
    logging.info(f"IDs of failed problems: {failed_problem_ids}")

def init_log(file=None, level=logging.INFO):
    format = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.getLogger().setLevel(level)
    formatter = logging.Formatter(format)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    stderr.setLevel(level)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(stderr)

    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=file, mode="w", encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logging.getLogger().addHandler(file_handler)

def load_deveval_dataset(file_path):
    problems = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "namespace": data["namespace"],
                "prompt": data["prompt"]
            })
    return problems

def extract_last_function_body(code_str: str) -> str:
    lines = code_str.splitlines()
    marker_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if '# The code to be completed is:' in ln:
            marker_idx = i
            break
    if marker_idx is None:
        return ""

    def_idx: Optional[int] = None
    def_pattern = re.compile(r'^\s*(async\s+def|def)\s')
    for i in range(marker_idx + 1, len(lines)):
        if def_pattern.match(lines[i]):
            def_idx = i
            break
    if def_idx is None:
        return ""

    leading_ws = re.match(r'^(\s*)', lines[def_idx]).group(1)
    def_indent = len(leading_ws.expandtabs(4))

    paren_balance = 0
    header_end = None
    for j in range(def_idx, len(lines)):
        ln = lines[j]
        paren_balance += ln.count('(') - ln.count(')')
        if paren_balance <= 0 and ln.rstrip().endswith(':'):
            header_end = j
            break
    if header_end is None:
        return ""

    body_start = header_end + 1
    first_body_line = None
    for k in range(body_start, len(lines)):
        if lines[k].strip() != "":
            first_body_line = k
            break
    if first_body_line is None:
        return ""

    first_body_indent = len(re.match(r'^(\s*)', lines[first_body_line]).group(1).expandtabs(4))
    if first_body_indent <= def_indent:
        return ""

    triple_re = re.compile(r'^\s*(?:[rubfRUBF]{0,3})("""|\'\'\')')
    m = triple_re.match(lines[first_body_line])
    content_start = first_body_line
    if m:
        delim = m.group(1)
        line_text = lines[first_body_line]
        if line_text.count(delim) >= 2:
            content_start = first_body_line + 1
        else:
            end_idx = None
            for t in range(first_body_line + 1, len(lines)):
                if delim in lines[t]:
                    end_idx = t
                    break
            if end_idx is None:
                return ""
            content_start = end_idx + 1

    collected = []
    for idx in range(content_start, len(lines)):
        ln = lines[idx]
        if ln.strip() == "":
            collected.append(ln)
            continue
        indent = len(re.match(r'^(\s*)', ln).group(1).expandtabs(4))
        if indent <= def_indent:
            break
        collected.append(ln)

    while collected and collected[0].strip() == "":
        collected.pop(0)
    while collected and collected[-1].strip() == "":
        collected.pop(-1)

    return "\n".join(collected)





parser = argparse.ArgumentParser()

parser.add_argument('--model', help='model name', required=True, type=str)
# parser.add_argument('--beam', help='beam size', required=False, type=int, default=1)
parser.add_argument('--decoding_mode', help='decoding mode, Traditional or AdaFixL', required=False, type=str, default='Traditional')
parser.add_argument('--entropy_threshold', help='entropy threshold, Learned or a number', required=False, type=str, default='Learned')

parser.add_argument('--max_new_tokens', help='max new tokens', required=False, type=int, default=512)
parser.add_argument('--lambda_value', help='lambda value', required=False, type=int, default=1)
parser.add_argument('--lookahead_length', help='lookahead length', required=False, type=int, default=5)
parser.add_argument('--lookahead_beam_size', help='lookahead beam size', required=False, type=int, default=3)
parser.add_argument('--logging_detail', help='logging detail or not', action='store_true')

parser.add_argument('--dataset', help='select a dataset, humaneval or mbpp', required=False, type=str)



def main():

    args = parser.parse_args()
    
    filename = f"experiments/{args.dataset}_outputs/{args.model}/{args.decoding_mode}/{args.model}"
    
    init_log(f"{filename}.log")
    
    if args.dataset == 'mbpp+':
        ds = load_dataset("evalplus/mbppplus")
        problems = ds["test"]
        
        print(f"Read {len(problems)} problems.")
        
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            beam_size=1,
            decoding_mode=args.decoding_mode,
            entropy_threshold=args.entropy_threshold
        )
        
        start_time = time.time()
        results = []

        for idx, problem in enumerate(problems):
            task_id = problem["task_id"]
            
            prompt = build_prompt(problem)
            if args.model.startswith('qwen3-'):
                prompt = '/no_think\n' + prompt

            # Log problem details
            logging.info(f"##### TASK_ID: {task_id} ######")
            logging.info(f"##### PROMPT ######\n{prompt}")
            logging.info(f"##### SOLUTION ######\n{problem['code']}")

            # Generate
            generated_code = generator.generate(
                prompt=prompt,
                beam_size=args.beam,
                max_new_tokens=args.max_new_tokens,
                lambda_value=args.lambda_value,
                lookahead_length=args.lookahead_length,
                lookahead_beam_size=args.lookahead_beam_size,
                logging_detail=args.logging_detail
            )
            
            for i, pred in enumerate(generated_code):
                generated_code[i] = prompt + pred

            if isinstance(generated_code, list):
                generated_code = [extract_python_code(pred) for pred in generated_code]
            else:
                generated_code = [extract_python_code(generated_code)]

            for i, pred in enumerate(generated_code):
                logging.info(f"##### PREDICTION-{i + 1} ######\n{pred}")

            results.append({
                "problem_id": task_id,
                "prompt": prompt,
                "generated_code": generated_code,
            })
            logging.info("\n\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

        for res, problem in tqdm(zip(results, problems), total=len(results), desc="Evaluating results"):
            generated_code = res["generated_code"][0] if isinstance(res["generated_code"], list) else res["generated_code"]
            eval_result = evaluate_generation_mbppplus(problem, generated_code)
            res["evaluation"] = eval_result

        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        summarize_results(results)
    
    elif args.dataset == 'humaneval+':
        ds = load_dataset("evalplus/humanevalplus")
        problems = ds["test"]
        
        print(f"Read {len(problems)} problems.")
        
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            beam_size=1,
            decoding_mode=args.decoding_mode,
            entropy_threshold=args.entropy_threshold
        )
        
        start_time = time.time()
        results = []

        for idx, problem in enumerate(problems):
            task_id = problem["task_id"]
            
            prompt = problem["prompt"]
            if args.model.startswith('qwen3-'):
                prompt = '/no_think\n' + prompt

            # Log problem details
            logging.info(f"##### TASK_ID: {task_id} ######")
            logging.info(f"##### PROMPT ######\n{prompt}")
            logging.info(f"##### SOLUTION ######\n{problem['canonical_solution']}")

            # Generate
            generated_code = generator.generate(
                prompt=prompt,
                beam_size=args.beam,
                max_new_tokens=args.max_new_tokens,
                lambda_value=args.lambda_value,
                lookahead_length=args.lookahead_length,
                lookahead_beam_size=args.lookahead_beam_size,
                logging_detail=args.logging_detail
            )
            
            for i, pred in enumerate(generated_code):
                generated_code[i] = prompt + pred

            if isinstance(generated_code, list):
                generated_code = [extract_python_code(pred) for pred in generated_code]
            else:
                generated_code = [extract_python_code(generated_code)]

            for i, pred in enumerate(generated_code):
                logging.info(f"##### PREDICTION-{i + 1} ######\n{pred}")

            results.append({
                "problem_id": task_id,
                "prompt": prompt,
                "generated_code": generated_code,
            })
            logging.info("\n\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

        for res, problem in tqdm(zip(results, problems), total=len(results), desc="Evaluating results"):
            generated_code = res["generated_code"][0] if isinstance(res["generated_code"], list) else res["generated_code"]
            eval_result = evaluate_generation_humanevalplus(problem, generated_code)
            res["evaluation"] = eval_result

        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        summarize_results(results)
        
    elif args.dataset == 'deveval':
        problems = load_deveval_dataset('data/deveval_filtered_data.jsonl')

        print(f"Read {len(problems)} problems.")
        
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator_DevEval(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            beam_size=1,
            decoding_mode=args.decoding_mode,
            entropy_threshold=args.entropy_threshold
        )
        
        start_time = time.time()
        results = []

        for idx, problem in enumerate(problems):
            task_id = idx
            
            prompt = problem["prompt"]
            if args.model.startswith('qwen3-'):
                prompt = '/no_think\n' + prompt

            # Log problem details
            logging.info(f"##### TASK_ID: {task_id} ######")
            logging.info(f"##### PROMPT ######\n{prompt}")

            # Generate
            generated_code = generator.generate(
                prompt=prompt,
                beam_size=args.beam,
                max_new_tokens=args.max_new_tokens,
                lambda_value=args.lambda_value,
                lookahead_length=args.lookahead_length,
                lookahead_beam_size=args.lookahead_beam_size,
                logging_detail=args.logging_detail
            )
            
            logging.info(f"##### ORIGIN_GENERATED_CODE ######\n{generated_code[0]}\n")
            
            for i, pred in enumerate(generated_code):
                generated_code[i] = prompt + pred

            if isinstance(generated_code, list):
                generated_code = [extract_last_function_body(pred) for pred in generated_code]
            else:
                generated_code = [extract_last_function_body(generated_code)]

            for i, pred in enumerate(generated_code):
                logging.info(f"##### PREDICTION-{i + 1} ######\n{pred}")

            results.append({
                "namespace": problem["namespace"],
                "completion": (generated_code[0] if isinstance(generated_code, list) else generated_code) or "",
            })
            logging.info("\n\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Total time taken: {elapsed_time:.2f} seconds")


        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    else:
        raise ValueError("dataset must be 'humaneval+', 'mbpp+' or 'deveval'.")
    
    
    
if __name__ == "__main__":
    main()