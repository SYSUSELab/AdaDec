import json
import argparse
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from llm.models import MODEL_FACTORY
from llm.generator import Generator



def read_problems_bigcodebench(filename="BigCodeBench-v0.1.1.jsonl"):
    problems = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    problem = json.loads(line.strip())
                    task_id = problem.get("task_id")
                    if task_id:
                        problems[task_id] = problem
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    continue
        return problems
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {str(e)}")



parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name', required=True, type=str)

if __name__ == "__main__":

    args = parser.parse_args()
    filename = f"data/gt_guide_data/{args.model}_statistics.parquet"

    problems = [(task_id, problem) for task_id, problem in read_problems_bigcodebench(filename='data/BigCodeBench-v0.1.1.jsonl').items()]
    print(f"Read {len(problems)} problems.")

    model, tokenizer = MODEL_FACTORY[args.model]()
    generator = Generator(model, tokenizer, model_name=args.model)
    samples = []
    for task_id, problem in tqdm(problems, ascii=True):
        prompt = problem["complete_prompt"]
        if args.model.startswith('qwen3-'):
            prompt = '/no_think\n' + prompt
        generator.generate_base_on_ground_truth(prompt=prompt,
                                                ground_truth=problem["canonical_solution"],
                                                filename=filename)
    print('Over.')

