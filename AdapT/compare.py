import json
import difflib
from typing import Dict, List

def compare_prompt_completion(jsonl_file_path: str, output_method: str = "console"):
    """
    从jsonl文件中逐行读取数据，比对prompt和completion字段的差异
    
    Args:
        jsonl_file_path: jsonl文件路径
        output_method: 输出方式，"console"控制台输出，"file"文件输出，"return"返回结果列表
    """
    results = []
    
    # 读取并处理每一行
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                if 'prompt' not in data or 'completion' not in data:
                    print(f"警告：第{line_num}行缺少prompt或completion字段")
                    continue
                    
                prompt = str(data['prompt'])
                completion = str(data['completion'])
                
                # 执行差异比较
                diff = list(difflib.ndiff(prompt.splitlines(), completion.splitlines()))
                
                # 构建结果对象
                result = {
                    'line_number': line_num,
                    'prompt': prompt,
                    'completion': completion,
                    'diff': diff,
                    'summary': {
                        'added_lines': sum(1 for d in diff if d.startswith('+ ')),
                        'removed_lines': sum(1 for d in diff if d.startswith('- ')),
                        'common_lines': sum(1 for d in diff if d.startswith('  '))
                    }
                }
                
                results.append(result)
                
                # 控制台输出
                if output_method == "console":
                    print(f"\n{'='*50}")
                    print(f"记录 #{line_num}")
                    print(f"{'='*50}")
                    print(f"Prompt ({len(prompt)}字符):")
                    print(f"{'-'*30}")
                    print(prompt)
                    print(f"\nCompletion ({len(completion)}字符):")
                    print(f"{'-'*30}")
                    print(completion)
                    print(f"\n差异分析:")
                    print(f"{'-'*30}")
                    for item in diff:
                        if item.startswith('- '):
                            print(f"\033[91m{item}\033[0m")  # 红色显示删除
                        elif item.startswith('+ '):
                            print(f"\033[92m{item}\033[0m")  # 绿色显示新增
                        else:
                            print(item)
                    print(f"\n统计: +{result['summary']['added_lines']}, -{result['summary']['removed_lines']}, 共{result['summary']['common_lines']}行相同")
                    
            except json.JSONDecodeError as e:
                print(f"JSON解析错误，第{line_num}行: {str(e)}")
                continue
            except Exception as e:
                print(f"处理第{line_num}行时发生错误: {str(e)}")
                continue
    
    # 处理输出方式
    if output_method == "file":
        output_file = jsonl_file_path.replace('.jsonl', '_diff.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"\n{'='*50}\n")
                f.write(f"记录 #{result['line_number']}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Prompt ({len(result['prompt'])}字符):\n{'-'*30}\n{result['prompt']}\n\n")
                f.write(f"Completion ({len(result['completion'])}字符):\n{'-'*30}\n{result['completion']}\n\n")
                f.write(f"差异分析:\n{'-'*30}\n")
                for item in result['diff']:
                    f.write(f"{item}\n")
                f.write(f"\n统计: +{result['summary']['added_lines']}, -{result['summary']['removed_lines']}, 共{result['summary']['common_lines']}行相同\n")
        print(f"差异结果已保存至: {output_file}")
    
    elif output_method == "return":
        return results

# 使用示例
if __name__ == "__main__":
    # 配置参数
    JSONL_FILE = "deveval_outputs/stabilityai-stable-code-instruct-3b.jsonl"  # 修改为你的文件路径
    
    # 执行比较（选择一种输出方式）
    # compare_prompt_completion(JSONL_FILE, output_method="console")
    
    # 或者保存到文件
    compare_prompt_completion(JSONL_FILE, output_method="file")
    
    # 或者获取结果列表进行进一步处理
    # results = compare_prompt_completion(JSONL_FILE, output_method="return")
    # for result in results:
    #     print(f"Line {result['line_number']}: {result['summary']}")
