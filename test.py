import re

def extract_function_body(code_string):
    """
    从字符串中提取函数体内容，保持原有缩进，去掉尾部第一个无缩进行及其之后的所有行
    """
    # 匹配函数定义行（def 开头），然后匹配文档字符串（可选），最后匹配函数体
    pattern = r'def\s+\w+\([^)]*\):\s*(?:\n\s*""".*?"""\s*)?(\n(?:\s{4,}.*\n?)*)'
    
    match = re.search(pattern, code_string, re.DOTALL)
    
    if match:
        function_body = match.group(1)
        # 去除开头的换行符，但保持内容的缩进
        function_body = function_body.lstrip('\n').rstrip()
        
        # 去掉尾部第一个无缩进行及其之后的所有行内容
        lines = function_body.split('\n')
        result_lines = []
        
        for line in lines:
            # 检查是否为无缩进行（非空行且不以空格开头）
            if line and not line.startswith(' '):
                # 遇到第一个无缩进行，停止添加
                break
            result_lines.append(line)
        
        return '\n'.join(result_lines).rstrip()
    
    return None

# 测试
s1 = '''
def is_json_serializable(val):
    """
    Check if the input value is JSON serializable. It checks if the input value is of the JSON serializable types.
    :param val: Any. The input value to be checked for JSON serializability.
    :return: Bool. True if the input value is JSON serializable, False otherwise.
    """
    try:
        json.dumps(val)
        return True
    except (TypeError, OverflowError):
        return False
'''

# 测试用例2：包含尾部无缩进行
s2 = '''
def test_function():
    x = 1
    y = 2
    return x + y

print("这是一个无缩进的行")
def another_function():
    pass
'''

result1 = extract_function_body(s1)
print("测试1 - 提取的函数体:")
print(repr(result1))
print("\n实际显示:")
print(result1)

result2 = extract_function_body(s2)
print("\n\n测试2 - 提取的函数体:")
print(repr(result2))
print("\n实际显示:")
print(result2)