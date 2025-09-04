import re
from typing import Optional

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

# -------------------------
# 简单测试（使用你给的示例）
if __name__ == "__main__":
    code_str = '''
def value_error(value, cls):
    value = repr(value)
    if len(value) > 50:
        value = value[:50] + "..."
    raise ValueError("Value '{}' can't be {}".format(value, cls.__name__))


class Field(object):
    """Base Field class - all fields should inherit from this

    As the fallback for all other field types are the BinaryField, this Field
    actually implements what is expected in the BinaryField
    """

    TYPE = (type(None),)

    @classmethod


# The code to be completed is:
    def serialize(cls, value, *args, **kwargs):

        """
        This function serializes a value to be exported. It should always return a unicode value, except for BinaryField.
        Input-Output Arguments
        :param cls: Class. The class instance.
        :param value: Any. The value to be serialized.
        :param *args: Tuple. Additional positional arguments.
        :param **kwargs: Dictionary. Additional keyword arguments.
        :return: Any. The serialized value.
        """
        raise NotImplementedError("This method is abstract and must be overridden in subclasses.")

    @classmethod
    '''
    res = extract_last_function_body(code_str)
    print("=== EXTRACTED ===")
    print(res)
    # 预期:
    #     raise NotImplementedError("This method is abstract and must be overridden in subclasses.")
