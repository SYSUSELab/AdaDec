import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import json
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from llm.models import MODEL_FACTORY
from llm.generator import Generator

try:
    from evalplus.data import (
        get_human_eval_plus,
        get_mbpp_plus,
        get_human_eval_plus_hash,
        get_mbpp_plus_hash,
    )
    from evalplus.evaluate import get_groundtruth, check_correctness
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.sanitize import sanitize
    from evalplus.eval import PASS
    from evalplus.data.utils import CACHE_DIR 
except ImportError as e:
    print("Error importing EvalPlus. Please run: pip install evalplus")
    sys.exit(1)

def init_log(file=None, level=logging.INFO):
    log_format = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.getLogger().setLevel(level)
    formatter = logging.Formatter(log_format)
    logging.getLogger().handlers = []
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    logging.getLogger().addHandler(stderr)
    if file:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        file_handler = logging.FileHandler(filename=file, mode="w", encoding='utf-8')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str, help='humaneval+ or mbpp+')
parser.add_argument('--decoding_mode', default='Traditional', type=str)
parser.add_argument('--max_new_tokens', default=1024, type=int)
parser.add_argument('--beam', default=1, type=int)
parser.add_argument('--logging_detail', help='log details', action='store_true')

parser.add_argument('--entropy_threshold', default='Learned', type=str)
parser.add_argument('--lambda_value', default=1.0, type=float)
parser.add_argument('--lookahead_length', default=5, type=int)
parser.add_argument('--lookahead_beam_size', default=3, type=int)

parser.add_argument('--dir', default="default", type=str)

def run_evaluation(args):
    is_chat_model = "instruct" in args.model.lower() or "chat" in args.model.lower()
    
    dataset_key = args.dataset.replace("+", "") # humaneval+ -> humaneval
    logging.info(f"Loading {dataset_key} dataset (Mini={args.mini})...")
    
    tasks_only_output_not_none = []

    if dataset_key == "humaneval":
        problems = get_human_eval_plus(mini=args.mini, noextreme=args.no_extreme)
        dataset_hash = get_human_eval_plus_hash(mini=args.mini, noextreme=args.no_extreme)
    elif dataset_key == "mbpp":
        problems = get_mbpp_plus(mini=args.mini, noextreme=args.no_extreme)
        dataset_hash = get_mbpp_plus_hash(mini=args.mini, noextreme=args.no_extreme)
        tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS
    else:
        raise ValueError("Dataset must be humaneval+ or mbpp+")

    try:
        expected_output = get_groundtruth(problems, dataset_hash, tasks_only_output_not_none)
    except (EOFError, pickle.UnpicklingError, TypeError) as e:
        logging.warning(f"GroundTruth cache corrupted ({e}). Removing and regenerating...")
        cache_file = os.path.join(CACHE_DIR, f"{dataset_hash}.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        expected_output = get_groundtruth(problems, dataset_hash, tasks_only_output_not_none)
    
    logging.info(f"Initializing {args.model}...")
    model_init_fn = MODEL_FACTORY.get(args.model)
    if not model_init_fn:
        raise ValueError(f"Model {args.model} not found in MODEL_FACTORY")
    
    model, tokenizer = model_init_fn()
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        beam_size=args.beam,
        decoding_mode=args.decoding_mode,
        entropy_threshold=args.entropy_threshold
    )

    logging.info("Starting generation...")
    generated_samples = []
    
    for task_id, task in tqdm(list(problems.items()), desc=f"Generating {dataset_key}"):
        prompt = task["prompt"]
        
        batch_completions = generator.generate(
            prompt=prompt,
            beam_size=args.beam,
            max_new_tokens=args.max_new_tokens,
            lambda_value=args.lambda_value,
            lookahead_length=args.lookahead_length,
            lookahead_beam_size=args.lookahead_beam_size,
            logging_detail=args.logging_detail
        )
        raw_completion = batch_completions[0]

        if is_chat_model:
            code_to_sanitize = raw_completion
        else:
            code_to_sanitize = prompt + raw_completion

        sanitized_code = sanitize(code=code_to_sanitize, entrypoint=task["entry_point"])
        
        generated_samples.append({
            "task_id": task_id,
            "solution": sanitized_code,
            "raw_completion": raw_completion
        })

    logging.info("Starting evaluation...")
    results = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = []
        for sample in generated_samples:
            task_id = sample["task_id"]
            task = problems[task_id]
            
            future = executor.submit(
                check_correctness,
                dataset=dataset_key,
                completion_id=0,
                problem=task,
                solution=sample["solution"],
                expected_output=expected_output[task_id],
                base_only=False,
                fast_check=True,
                identifier=task_id,
                min_time_limit=30,
                gt_time_limit_factor=4.0
            )
            futures.append((future, sample))

        for future, sample in tqdm(futures, desc="Evaluating"):
            eval_result = future.result()
            base_status = eval_result["base"][0]
            plus_status = eval_result["plus"][0]
            
            results.append({
                "task_id": sample["task_id"],
                "solution": sample["solution"],
                "raw_completion": sample["raw_completion"],
                "base_status": base_status,
                "plus_status": plus_status
            })

    total = len(results)
    base_pass_count = sum(1 for r in results if r["base_status"] == PASS)
    
    plus_pass_count = sum(1 for r in results if r["base_status"] == PASS and r["plus_status"] == PASS)

    base_score = base_pass_count / total * 100 if total > 0 else 0
    plus_score = plus_pass_count / total * 100 if total > 0 else 0

    logging.info(f"====== Results for {args.model} ({args.decoding_mode}) ======")
    logging.info(f"Tasks: {total}")
    logging.info(f"Base Pass@1: {base_score:.2f}% ({base_pass_count}/{total})")
    logging.info(f"Plus Pass@1: {plus_score:.2f}% ({plus_pass_count}/{total})")

    output_dir = f"experiments/{args.dataset}_outputs/{args.model}/{args.decoding_mode}/{args.dir}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.jsonl")
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

def main():
    args = parser.parse_args()
    
    base_path = f"experiments/{args.dataset}_outputs/{args.model}/{args.decoding_mode}"
    target_dir = os.path.join(base_path, args.dir)

    while os.path.exists(target_dir):
        args.dir = args.dir + "_new"
        target_dir = os.path.join(base_path, args.dir)
    
    print(f"Output directory exists. Switching to new directory: {target_dir}")

    log_file = os.path.join(target_dir, "eval.log")
    
    os.makedirs(target_dir, exist_ok=True)
    
    init_log(log_file)
    run_evaluation(args)

if __name__ == "__main__":
    main()
