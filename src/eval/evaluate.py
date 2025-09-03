import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import argparse
import logging
import re
import sys
from pathlib import Path
import time
import json
import multiprocessing
from typing import List
from datasets import load_dataset
import traceback

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from llm.models import MODEL_FACTORY
from llm.generator import Generator
from llm.generator_deveval import Generator_DevEval

from data.human_eval.human_eval.data import write_jsonl, read_problems, stream_jsonl, HUMAN_EVAL
from data.human_eval.human_eval.evaluation import evaluate_functional_correctness



def load_mbpp_dataset(file_path, num):
    """Load the MBPP dataset from a JSONL file."""
    i = 0
    problems = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if i < num:
                problems.append(json.loads(line))
                i += 1
            else:
                break
    return problems

def extract_code_lines(jsonl_file: str) -> List[str]:
    results = []
    def_name_pattern = re.compile(r'^def\s+(\w+)\s*\(')
    assert_func_pattern = re.compile(r'assert\s+(\w+)\s*\(')

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {idx}: JSON decode error - {e}")

            if 'code' not in entry or 'test_list' not in entry:
                raise KeyError(f"Line {idx}: Missing 'code' or 'test_list' field")

            code = entry['code']
            test_list = entry['test_list']

            code_lines = code.splitlines()
            unindented_lines = []
            def_lines = []

            for cl in code_lines:
                if cl.startswith('def '):
                    def_lines.append(cl)
                elif not cl.startswith((' ', '\t')) and cl.strip():
                    unindented_lines.append(cl)

            selected_defs = []
            if len(def_lines) == 1:
                selected_defs = def_lines
            elif len(def_lines) > 1:
                func_names = []
                for test in test_list:
                    for m in assert_func_pattern.finditer(test):
                        func_names.append(m.group(1))
                func_names = list(dict.fromkeys(func_names))
                for dl in def_lines:
                    m = def_name_pattern.match(dl)
                    if m and m.group(1) in func_names:
                        selected_defs.append(dl)
                if not selected_defs:
                    selected_defs = def_lines

            extracted = unindented_lines + selected_defs
            results.append('\n'.join(extracted))

    return results

def make_prompt(prompt_head: str, text: str) -> str:
    doc = '    """\n    ' + text + '\n    """\n'
    return f"{prompt_head}\n{doc}"

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

def _run_code_with_tests(code, test_list, result_dict):
    """
    Helper function to run in a separate process.
    Executes the code and runs all tests. Stores the result (True/False) in result_dict.
    """
    namespace = {}
    try:
        exec(code, namespace)
        func_name = test_list[0].split("(")[0].split()[-1]
        if func_name not in namespace:
            result_dict['passed'] = False
            return
        for test in test_list:
            exec(test, namespace)
        result_dict['passed'] = True
    except Exception as e:
        result_dict['passed'] = False

def evaluate_generation(problem, generated_code, timeout=3):
    """
    Evaluate the first generated code prediction using pass@1 metric.
    Returns True if the first prediction passes all tests, False otherwise.
    Timeout is in seconds.
    """
    code_to_eval = generated_code[0] if isinstance(generated_code, list) else generated_code

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=_run_code_with_tests,
        args=(code_to_eval, problem["test_list"], result_dict)
    )
    p.start()
    p.join(timeout)

    # Timeout
    if p.is_alive():
        p.terminate()
        p.join()
        logging.warning(f"Execution timed out for problem {problem['task_id']}")
        return False

    return result_dict.get('passed', False)

def build_prompt(example: dict) -> str:
    """
    从 mbppplus 的一条样本构造 prompt:
    [可能的 import 代码] + def 函数签名 + docstring(来自prompt) + # YOUR CODE HERE
    """
    code = example["code"].strip().splitlines()

    def_index = None
    for i, line in enumerate(code):
        if line.strip().startswith("def "):
            def_index = i
            break
    if def_index is None:
        raise ValueError(f"无法在 code 中找到函数定义: {example['code'][:80]}")

    # 前面的 import / 辅助代码
    prefix_code = "\n".join(code[:def_index])

    # 函数定义行
    def_line = code[def_index]

    # 题目描述放在 docstring
    docstring = example["prompt"].strip()

    # 组装
    prompt_parts = []
    if prefix_code:
        prompt_parts.append(prefix_code)  # import 语句
    prompt_parts.append(f"""{def_line}
    \"\"\" {docstring} \"\"\"
    # YOUR CODE HERE""")

    return "\n".join(prompt_parts) + "\n"


import traceback
import multiprocessing

def _exec_with_env(code: str, env: dict, problem: dict):
    """子进程中执行代码 + 测试"""
    # 执行模型生成的代码
    exec(code, env, env)

    # 执行 test_imports
    for imp in problem.get("test_imports", []):
        exec(imp, env, env)

    # 执行 test_list 的断言
    for t in problem.get("test_list", []):
        exec(t, env, env)

    # 执行完整的 test 脚本
    exec(problem["test"], env, env)


import multiprocessing
import traceback

def evaluate_generation_mbppplus(problem: dict, generated_code: str, timeout: int = 5):
    """
    针对 mbppplus 的简化评估函数（带超时机制）
    :param problem: 一条 mbppplus 的数据
    :param generated_code: 模型生成的代码 (str)
    :param timeout: 最大允许执行秒数
    :return: dict，只包含 evaluation 字段 (True/False)
    """
    evaluation = False  # 默认不通过

    # 每个候选代码独立环境
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





# HumanEvalPlus

import multiprocessing
import traceback
import builtins

def exec_with_env_humanevalplus(code: str, env: dict, problem: dict):
    """子进程中执行代码 + 测试"""
    # 执行模型生成的代码
    exec(code, env, env)
    
    # 执行完整的 test 脚本
    exec(problem["test"], env, env)
    
    # 获取函数名并调用 check 函数进行测试
    entry_point = problem["entry_point"]
    if entry_point in env:
        env["check"](env[entry_point])
    else:
        raise NameError(f"Function '{entry_point}' not found in generated code")

def evaluate_generation_humanevalplus(problem: dict, generated_code: str, timeout: int = 5):
    """
    针对 HumanEval+ 的简化评估函数（带超时机制）
    :param problem: 一条 HumanEval+ 的数据
    :param generated_code: 模型生成的代码 (str)
    :param timeout: 最大允许执行秒数
    :return: bool，表示是否通过所有测试
    """
    evaluation = False  # 默认不通过
    
    # 每个候选代码独立环境
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



def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    logging.info(results)

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

def extract_first_func(code):
    lines = code.split("\n")
    while len(lines) > 1 and lines[0].strip() == "":
        lines = lines[1:]

    if not lines:
        return ""

    indent = 0
    indent_mobj = re.search("^\s*", lines[0])
    if indent_mobj:
        indent = len(indent_mobj.group(0))

    comment = ""
    if lines[0].startswith(" " * indent + "'''"):
        while len(lines) > 0:
            line = lines.pop(0)
            comment += line + "\n"
            if line.strip().endswith("'''"):
                break
    elif lines[0].startswith(" " * indent + '"""'):
        while len(lines) > 0:
            line = lines.pop(0)
            comment += line + "\n"
            if line.strip().endswith('"""'):
                break
    while len(lines) > 0:
        if lines[0].startswith(" " * indent + "#"):
            line = lines.pop(0)
            comment += line + "\n"
        else:
            break

    if not lines:
        return comment

    new_lines = [lines[0]]
    for line in lines[1:]:
        if indent > 0 and re.match(r" {0,%d}[^\s#]" % (indent - 1), line):
            break
        new_lines.append(line)

    func = "\n".join(new_lines)
    func = comment + func
    return func



def load_deveval_dataset(file_path):
    """
    从jsonl文件中读取DevEval数据集，返回列表 problems。
    每一条数据包含 namespace 和 prompt 两个字段。
    """
    problems = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            problems.append({
                "namespace": data["namespace"],
                "prompt": data["prompt"]
            })
    return problems[500:]  # 只评估500之后的部分

def extract_python_code_deveval(text):
    """
    从字符串中提取Python代码块中的函数体内容
    
    Args:
        text (str): 包含Python代码的字符串
        
    Returns:
        str: 提取出的Python代码，保持原始缩进
    """
    # 查找Python代码块（用```python或```Python包围的部分）
    code_block_pattern = r'```[Pp]ython\n(.*?)\n```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if not matches:
        return ""
    
    # 取第一个匹配的代码块
    code_block = matches[0]
    
    # 按行分割代码
    lines = code_block.split('\n')
    
    # 找到函数定义行的索引
    def_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            def_line_index = i
            break
    
    if def_line_index == -1:
        # 如果没有找到函数定义，返回整个代码块（去掉首尾空行）
        result_lines = []
        for line in lines:
            if line.strip():  # 非空行
                result_lines.append(line)
        return '\n'.join(result_lines)
    
    # 从函数定义行之后开始处理
    remaining_lines = lines[def_line_index + 1:]
    
    # 跳过文档字符串
    in_docstring = False
    docstring_quotes = None
    start_index = 0
    
    for i, line in enumerate(remaining_lines):
        stripped_line = line.strip()
        
        if not in_docstring:
            # 检查是否是文档字符串的开始
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                in_docstring = True
                docstring_quotes = stripped_line[:3]
                # 检查是否在同一行结束
                if stripped_line.count(docstring_quotes) >= 2:
                    in_docstring = False
                    start_index = i + 1
                continue
            elif stripped_line and not stripped_line.startswith('#'):
                # 找到第一个非空非注释行，开始提取
                start_index = i
                break
        else:
            # 在文档字符串中，查找结束标记
            if docstring_quotes in line:
                in_docstring = False
                start_index = i + 1
    
    # 提取函数体内容
    function_body_lines = remaining_lines[start_index:]
    
    # 去掉末尾的空行
    while function_body_lines and not function_body_lines[-1].strip():
        function_body_lines.pop()
    
    # 如果没有函数体内容，返回空字符串
    if not function_body_lines:
        return ""
    
    # 直接保持原始缩进，不做调整
    result_lines = function_body_lines
    
    # 去掉结尾的空行
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines)

def extract_last_function_body(code_str: str) -> str:
    """
    提取代码字符串中最后一个 def 函数的函数体（基于 docstring 定位）。
    - 假设每个函数一定有 docstring。
    - 忽略 docstring，不包含在函数体中。
    - 保留函数体原始缩进。
    """
    lines: List[str] = code_str.splitlines()
    n = len(lines)

    # 找到最后一个 def
    def_start = None
    for i in range(n - 1, -1, -1):
        if re.match(r'^\s*def\s+', lines[i]):
            def_start = i
            break
    if def_start is None:
        return ""

    # 找到函数体的第一个三引号 (""" 或 ''')
    doc_start = None
    doc_delim = None
    for i in range(def_start, n):
        stripped = lines[i].lstrip()
        if stripped.startswith(('"""', "'''")):
            doc_start = i
            doc_delim = stripped[:3]
            break
    if doc_start is None:
        return ""  # 没有 docstring

    # 找到 docstring 结束行
    doc_end = doc_start
    if lines[doc_start].lstrip().count(doc_delim) >= 2:  
        # 单行 docstring
        pass
    else:
        doc_end += 1
        while doc_end < n:
            if doc_delim in lines[doc_end]:
                break
            doc_end += 1

    # 函数体从 doc_end+1 行开始
    body_start = doc_end + 1
    if body_start >= n:
        return ""

    # 计算函数体缩进（第一个非空行的缩进）
    def leading_indent_len(s: str) -> int:
        m = re.match(r'^(\s*)', s)
        return len(m.group(1).expandtabs(8)) if m else 0

    first_nonempty = None
    for i in range(body_start, n):
        if lines[i].strip():
            first_nonempty = i
            break
    if first_nonempty is None:
        return ""

    base_indent_len = leading_indent_len(lines[first_nonempty])

    # 收集函数体
    body_lines: List[str] = []
    for i in range(first_nonempty, n):
        line = lines[i]
        if line.strip() and leading_indent_len(line) < base_indent_len:
            break
        body_lines.append(line)

    # 去掉前后空行
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    return "\n".join(body_lines)




parser = argparse.ArgumentParser()

parser.add_argument('--model', help='model name', required=True, type=str)
parser.add_argument('--beam', help='beam size', required=False, type=int, default=1)
parser.add_argument('--decoding_mode', help='decoding mode, Traditional or AdaFixL or AdaDynL', required=False, type=str, default='Traditional')
parser.add_argument('--entropy_threshold', help='entropy threshold, Learned or a number', required=False, type=str, default='Learned')

parser.add_argument('--max_new_tokens', help='max new tokens', required=False, type=int, default=512)
parser.add_argument('--lambda_value', help='lambda value', required=False, type=int, default=1)
parser.add_argument('--lookahead_length', help='lookahead length', required=False, type=int, default=5)
parser.add_argument('--lookahead_beam_size', help='lookahead beam size', required=False, type=int, default=3)
parser.add_argument('--logging_detail', help='logging detail or not', action='store_true')

parser.add_argument('--dataset', help='select a dataset, humaneval or mbpp', required=False, type=str)
parser.add_argument('--dirname', help='directory name', required=False, type=str, default='new')



def main():

    args = parser.parse_args()
    
    filename = ""
    if args.beam > 1:
        filename = f"experiments/{args.dataset}_outputs/{args.model}/beamsearch_{args.beam}/{args.model}"
    else:
        filename = f"experiments/{args.dataset}_outputs/{args.model}/{args.decoding_mode}/{args.model}"
    
    init_log(f"{filename}.log")

    if args.dataset == 'humaneval':
        problems = [(task_id, problem) for task_id, problem in read_problems().items()]
        print(f"Read {len(problems)} problems.")

        start_time = time.time()  # Start timer
        
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator(model=model,
                              tokenizer=tokenizer, 
                              model_name=args.model,
                              beam_size=args.beam, 
                              decoding_mode=args.decoding_mode,
                              entropy_threshold=args.entropy_threshold)
            
        samples = []
        for task_id, problem in tqdm(problems, ascii=True):
            # disable the thinking mode
            if args.model.startswith('qwen3-') and 'prompt' in problem and isinstance(problem['prompt'], str):
                problem['prompt'] = '/no_think\n' + problem['prompt']
            
            logging.info(f"##### {task_id} ######")
            logging.info(f"##### PROMPT ######\n{problem['prompt']}")
            logging.info(f"##### SOLUTION ######\n{problem['canonical_solution']}")

            preds = generator.generate(prompt=problem["prompt"], 
                                       beam_size=args.beam, 
                                       max_new_tokens=args.max_new_tokens,
                                       lambda_value=args.lambda_value,
                                       lookahead_length=args.lookahead_length, 
                                       lookahead_beam_size=args.lookahead_beam_size, 
                                       logging_detail=args.logging_detail)
            
            preds = [extract_first_func(pred) for pred in preds]
            for i, pred in enumerate(preds):
                logging.info(f"##### PREDICTION-{i+1} ######\n{pred}")
            samples.append(dict(task_id=task_id, completion=preds[0]))

            logging.info("")
            logging.info("")
            logging.info("")
            logging.info("")

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

        write_jsonl(f"{filename}.jsonl", samples)
        entry_point(sample_file=f"{filename}.jsonl")
    
    elif args.dataset == 'mbpp':
        # Load MBPP dataset
        problems = load_mbpp_dataset('data/mbpp.jsonl', num=200)
        prompt_heads = extract_code_lines('data/mbpp.jsonl')
        
        print(f"Read {len(problems)} problems.")
        
        start_time = time.time()  # Start timere

        model, tokenizer = MODEL_FACTORY[args.model]()

        generator = Generator(model=model, 
                              tokenizer=tokenizer, 
                              model_name=args.model, 
                              beam_size=args.beam, 
                              decoding_mode=args.decoding_mode,
                              entropy_threshold=args.entropy_threshold)

        results = []
        for idx, problem in enumerate(problems):
            text = problem["text"]
            prompt_head = prompt_heads[idx]

            # disable the thinking mode
            if args.model.startswith('qwen3-'):
                prompt_head = '/no_think\n' + prompt_head
            

            prompt = make_prompt(prompt_head, text)
            
            # Log problem details
            logging.info(f"##### TASK_ID: {problem['task_id']} ######")
            logging.info(f"##### PROMPT ######\n{prompt}")
            logging.info(f"##### SOLUTION ######\n{problem['code']}")

            # Generate code
            generated_code = generator.generate(prompt=prompt, 
                                                beam_size=args.beam, 
                                                max_new_tokens=args.max_new_tokens,
                                                lambda_value=args.lambda_value,
                                                lookahead_length=args.lookahead_length, 
                                                lookahead_beam_size=args.lookahead_beam_size, 
                                                logging_detail=args.logging_detail)
            
            for i, pred in enumerate(generated_code):
                generated_code[i] = prompt_head + '\n' + pred

            if isinstance(generated_code, list):
                for i, pred in enumerate(generated_code):
                    generated_code[i] = extract_python_code(pred)
            else:
                generated_code = extract_python_code(generated_code)

            if isinstance(generated_code, list):
                for i, pred in enumerate(generated_code):
                    logging.info(f"##### PREDICTION-{i + 1} ######\n{pred}")
            else:
                logging.info(f"##### PREDICTION ######\n{generated_code}")

            results.append({
                "problem_id": problem["task_id"],
                "prompt": prompt,
                "generated_code": generated_code,
            })

            logging.info("")
            logging.info("")
            logging.info("")
            logging.info("")
            
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

        for res, problem in tqdm(zip(results, problems), total=len(results), desc="Evaluating results"):
            eval_result = evaluate_generation(problem, res["generated_code"])
            res["evaluation"] = eval_result

        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        summarize_results(results)

    elif args.dataset == 'mbpp+':
        # ========== Step 1. 加载数据集 ==========
        ds = load_dataset("evalplus/mbppplus")
        problems = ds["test"]  # 378 道题
        # 取前3个数据
        # problems = problems.select(range(3))
        
        print(f"Read {len(problems)} problems.")
        
        # ========== Step 2. 初始化模型 ==========
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            beam_size=args.beam,
            decoding_mode=args.decoding_mode,
            entropy_threshold=args.entropy_threshold
        )
        
        # ========== Step 3. 遍历生成 ==========
        start_time = time.time()
        results = []

        for idx, problem in enumerate(problems):
            # prompt 直接来自字段
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

            # 后处理
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

        # ========== Step 4. 评估 ==========
        for res, problem in tqdm(zip(results, problems), total=len(results), desc="Evaluating results"):
            generated_code = res["generated_code"][0] if isinstance(res["generated_code"], list) else res["generated_code"]
            eval_result = evaluate_generation_mbppplus(problem, generated_code)
            res["evaluation"] = eval_result

        # ========== Step 5. 保存结果 ==========
        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        summarize_results(results)
    
    elif args.dataset == 'humaneval+':
        # ========== Step 1. 加载数据集 ==========
        ds = load_dataset("evalplus/humanevalplus")
        problems = ds["test"] # 164 道题
        # 取前几个数据
        # problems = problems.select(range(20))
        
        print(f"Read {len(problems)} problems.")
        
        # ========== Step 2. 初始化模型 ==========
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            beam_size=args.beam,
            decoding_mode=args.decoding_mode,
            entropy_threshold=args.entropy_threshold
        )
        
        # ========== Step 3. 遍历生成 ==========
        start_time = time.time()
        results = []

        for idx, problem in enumerate(problems):
            # prompt 直接来自字段
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

            # 后处理
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

        # ========== Step 4. 评估 ==========
        for res, problem in tqdm(zip(results, problems), total=len(results), desc="Evaluating results"):
            generated_code = res["generated_code"][0] if isinstance(res["generated_code"], list) else res["generated_code"]
            eval_result = evaluate_generation_humanevalplus(problem, generated_code)
            res["evaluation"] = eval_result

        # ========== Step 5. 保存结果 ==========
        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        summarize_results(results)
        
    
    elif args.dataset == 'deveval':
        # ========== Step 1. 加载数据集 ==========
        problems = load_deveval_dataset('deveval_data.jsonl')

        # print(problems[0]["prompt"])
        # return
        
        print(f"Read {len(problems)} problems.")
        
        # ========== Step 2. 初始化模型 ==========
        model, tokenizer = MODEL_FACTORY[args.model]()
        generator = Generator_DevEval(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            beam_size=args.beam,
            decoding_mode=args.decoding_mode,
            entropy_threshold=args.entropy_threshold
        )
        
        # return
        
        # ========== Step 3. 遍历生成 ==========
        start_time = time.time()
        results = []

        for idx, problem in enumerate(problems):
            task_id = idx
            
            # prompt 直接来自字段
            prompt = problem["prompt"]
            if args.model.startswith('qwen3-'):
                prompt = '/no_think\n' + prompt

            # Log problem details
            logging.info(f"##### TASK_ID: {task_id} ######")
            logging.info(f"##### PROMPT ######\n{prompt}")
            # logging.info(f"##### SOLUTION ######\n{problem['canonical_solution']}")

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


        # ========== Step 5. 保存结果 ==========
        output_file = f"{filename}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        
    
    else:
        raise ValueError("dataset must be 'humaneval' or 'mbpp' or 'humaneval+' or 'mbpp+' or 'deveval'.")
    
    
    
if __name__ == "__main__":
    main()