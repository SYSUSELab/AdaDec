import re
from typing import Optional
import json



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


"""

从AdapT/deveval_outputs/deepseek-ai-deepseek-coder-1.3b-instruct.jsonl，提取每个json的original_code字段，

调用extract_last_function_body，将结果保存为新的completion字段。

将新的completion和原来jsonl中的namespace字段一起保存为新的jsonl文件。

"""


# json1 = {"namespace": "bentoml._internal.runner.container.DefaultContainer.batch_to_payloads", "completion": "", "original_code": "# Please complete the get_worker_env function based on the contexts above the function.\n\n# The contexts above the function are:\nfrom __future__ import annotations\n\nimport abc\nimport logging\nimport math\nimport typing as t\n\nfrom ..resource import get_resource\nfrom ..resource import system_resources\nfrom .runnable import Runnable\n\nlogger = logging.getLogger(__name__)\n\n\nclass Strategy(abc.ABC):\n    @classmethod\n    @abc.abstractmethod\n    def get_worker_count(\n        cls,\n        runnable_class: t.Type[Runnable],\n        resource_request: dict[str, t.Any] | None,\n        workers_per_resource: int | float,\n    ) -> int:\n        ...\n\n    @classmethod\n    @abc.abstractmethod\n    def get_worker_env(\n        cls,\n        runnable_class: t.Type[Runnable],\n        resource_request: dict[str, t.Any] | None,\n        workers_per_resource: int | float,\n        worker_index: int,\n    ) -> dict[str, t.Any]:\n        \"\"\"\n        Args:\n            runnable_class : The runnable class to be run.\n            resource_request : The resource request of the runnable.\n            worker_index : The index of the worker, start from 0.\n        \"\"\"\n        ...\n\n\nTHREAD_ENVS = [\n    \"BENTOML_NUM_THREAD\",  # For custom Runner code\n    \"OMP_NUM_THREADS\",  # openmp\n    \"OPENBLAS_NUM_THREADS\",  # openblas,\n    \"MKL_NUM_THREADS\",  # mkl,\n    \"VECLIB_MAXIMUM_THREADS\",  # accelerate,\n    \"NUMEXPR_NUM_THREADS\",  # numexpr\n    # For huggingface fast tokenizer\n    \"RAYON_RS_NUM_CPUS\",\n    # For Tensorflow\n    \"TF_NUM_INTEROP_THREADS\",\n    \"TF_NUM_INTRAOP_THREADS\",\n]  # TODO(jiang): make it configurable?\n\n\nclass DefaultStrategy(Strategy):\n    @classmethod\n    def get_worker_count(\n        cls,\n        runnable_class: t.Type[Runnable],\n        resource_request: dict[str, t.Any] | None,\n        workers_per_resource: int | float,\n    ) -> int:\n        if resource_request is None:\n            resource_request = system_resources()\n\n        # use nvidia gpu\n        nvidia_gpus = get_resource(resource_request, \"nvidia.com/gpu\")\n        if (\n            nvidia_gpus is not None\n            and len(nvidia_gpus) > 0\n            and \"nvidia.com/gpu\" in runnable_class.SUPPORTED_RESOURCES\n        ):\n            return math.ceil(len(nvidia_gpus) * workers_per_resource)\n\n        # use CPU\n        cpus = get_resource(resource_request, \"cpu\")\n        if cpus is not None and cpus > 0:\n            if \"cpu\" not in runnable_class.SUPPORTED_RESOURCES:\n                logger.warning(\n                    \"No known supported resource available for %s, falling back to using CPU.\",\n                    runnable_class,\n                )\n\n            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:\n                if isinstance(workers_per_resource, float):\n                    raise ValueError(\n                        \"Fractional CPU multi threading support is not yet supported.\"\n                    )\n                return workers_per_resource\n\n            return math.ceil(cpus) * workers_per_resource\n\n        # this should not be reached by user since we always read system resource as default\n        raise ValueError(\n            f\"No known supported resource available for {runnable_class}. Please check your resource request. \"\n            \"Leaving it blank will allow BentoML to use system resources.\"\n        )\n\n    @classmethod\n\n\n# The code to be completed is:\n    def get_worker_env(\n        cls,\n        runnable_class: t.Type[Runnable],\n        resource_request: dict[str, t.Any] | None,\n        workers_per_resource: int | float,\n        worker_index: int,\n    ) -> dict[str, t.Any]:\n\n        \"\"\"\n        This function is a method of the DefaultStrategy class. It is used to get the environment variables for a worker process based on the given parameters. It determines whether to use GPU or CPU based on the resource request and the runnable class. It sets the appropriate environment variables accordingly.\n        Input-Output Arguments\n        :param cls: DefaultStrategy. The class itself.\n        :param runnable_class: Type[Runnable]. The class of the runnable to be executed.\n        :param resource_request: dict[str, t.Any] | None. The resource request of the runnable. Defaults to None.\n        :param workers_per_resource: int | float. The number of workers per resource. Defaults to None.\n        :param worker_index: int. The index of the worker. Starts from 0.\n        :return: dict[str, t.Any]. The environment variables for the worker process.\n        \"\"\"\n        # Get the worker count\n        worker_count = cls.get_worker_count(runnable_class, resource_request, workers_per_resource)\n\n        # Set the environment variables\n        env = {}\n        for i, thread_env in enumerate(THREAD_ENVS):\n            env[thread_env] = str(i)\n\n        return env\n\n# Please complete the get_worker_env function based on the contexts above the function.\n# The contexts above the function are:\nfrom __future__ import annotations\n\nimport abc\nimport logging\nimport math\nimport typing as t\n\nfrom ..resource import get_resource\nfrom ..resource import system_resources\nfrom .runnable import Runnable\n\nlogger = logging.getLogger(__name__)\n\n\nclass Strategy(abc.ABC):\n    @classmethod\n    @abc.abstractmethod\n    def get_worker_count(\n        cls,\n        runnable_class: t.Type[Runnable],\n        resource_request: dict[str, t.Any] | None,\n        workers_per_resource: int | float,\n    ) -> int:\n        ...\n\n    @classmethod\n    @abc.abstractmethod\n    def get_worker_env(\n        cls,\n        runnable_class: t.Type[Runnable],\n        resource_request: dict[str, t.Any] | None,\n        workers_per_resource: int | float,\n        worker_index: int,\n    ) -> dict[str, t.Any]:\n        \"\"\"\n        Args:\n            runnable_class : The runnable class to be run.\n            resource_request : The resource request of the runnable.\n            worker_index : The index of the worker, start from 0.\n        \"\"\"\n        ...\n\n\nTHREAD_ENVS = [\n    \"BENTOML_NUM_THREAD\",  # For custom Runner code\n    \"OMP_NUM_THREADS\",  # openmp\n    \"OPENBLAS_NUM_THREADS\",  # openblas,\n    \"MKL_NUM_THREADS\",  # mkl"}

# print(extract_last_function_body(json1["original_code"]))



def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # 如果某一行不是有效 JSON，跳过
                continue

            # 提取 original_code
            original_code = data.get("original_code", "")
            # 提取 namespace
            namespace = data.get("namespace", "")

            # 提取最后一个函数体
            new_completion = extract_last_function_body(original_code)

            # 构建新的 JSON 对象
            new_entry = {
                "namespace": namespace,
                "completion": new_completion
            }

            # 写入新的 JSONL 文件
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

    print(f"处理完成，结果已保存至 {output_file}")

# 主程序
if __name__ == "__main__":
    input_path = "deveval_outputs/deepseek-ai-deepseek-coder-1.3b-instruct.jsonl"
    output_path = "extracted_completions.jsonl"  # 输出文件名，可根据需要修改
    process_jsonl(input_path, output_path)