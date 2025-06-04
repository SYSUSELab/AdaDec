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
from typing import List

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from llm.models import MODEL_FACTORY
from llm.generator import Generator

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



if __name__ == "__main__":

    args = parser.parse_args()
    
    filename = f"experiments/{args.dataset}_outputs/{args.model}/{args.dirname}/{args.model}"
    
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
                generated_code[0] = extract_python_code(generated_code)

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
    
    else:
        raise ValueError("dataset must be 'humaneval' or 'mbpp'.")