import multiprocessing
import traceback
import builtins
from tqdm import tqdm
import json
import argparse



def load_dataset(file_path):
    """
    从 JSONL 文件中加载数据集。

    参数:
        file_path (str): JSONL 文件的路径。

    返回:
        list: 包含每行 JSON 对象的列表，每个元素是一个 dict。
    """
    problems = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                problem = json.loads(line)
                problems.append(problem)
    return problems

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

def summarize_results(results, output_file=None):
    total_problems = len(results)
    passed_problems = sum(1 for res in results if res.get("evaluation"))
    pass_rate = (passed_problems / total_problems) * 100 if total_problems > 0 else 0
    failed_problem_ids = [res["problem_id"] for res in results if not res.get("evaluation")]

    summary_lines = [
        f"Total number of problems: {total_problems}",
        f"Number of problems passed: {passed_problems}",
        f"Pass rate: {pass_rate:.2f}%",
        f"IDs of failed problems: {failed_problem_ids}"
    ]

    # Print to console
    for line in summary_lines:
        print(line)

    # Optionally write to result file
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            for line in summary_lines:
                f.write(line + "\n")

def load_completions(file_path):
    """
    从 JSONL 文件中提取所有 'completion' 字段，并返回列表。

    参数:
        file_path (str): JSONL 文件的路径。

    返回:
        List[str]: 包含所有 completion 内容的列表。
                   如果某行没有 'completion' 字段或解析失败，则跳过该行。
    """
    completions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            try:
                data = json.loads(line)
                completion = data.get("completion")
                if completion is not None:
                    completions.append(completion)
                else:
                    print(f"警告: 第 {line_num} 行缺少 'completion' 字段，已跳过。")
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败，错误: {e}，已跳过。")
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理时发生未知错误: {e}，已跳过。")
    
    return completions



def main(completions_file, result_file):
    problems = load_dataset("inputs/HumanEval.jsonl")

    results = []

    generated_codes = load_completions(completions_file)

    for idx, problem in enumerate(problems):
        # prompt 直接来自字段
        task_id = problem["task_id"]
        
        prompt = problem["prompt"]
        if 'Qwen3' in completions_file:
            prompt = '/no_think\n' + prompt
        
        generated_code = generated_codes[idx]
        
        generated_code = prompt + generated_code

        results.append({
            "problem_id": task_id,
            "prompt": prompt,
            "generated_code": generated_code,
        })

    for res, problem in tqdm(zip(results, problems), total=len(results), desc="Evaluating results"):
        generated_code = res["generated_code"][0] if isinstance(res["generated_code"], list) else res["generated_code"]
        eval_result = evaluate_generation_humanevalplus(problem, generated_code)
        res["evaluation"] = eval_result

    output_file = result_file
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    summarize_results(results, output_file=result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated code completions.")
    parser.add_argument("--completion_file", type=str, required=True, help="Path to the completion.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the result.")

    args = parser.parse_args()
    
    main(args.completion_file, args.result_file)