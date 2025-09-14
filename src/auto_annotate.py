import ast
import itertools
from typing import Optional, Tuple
import re

class AlphaRenamer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.counter = itertools.count()
        self.env = {}  # {original name: new name}

    def fresh(self) -> str:
        return f"v{next(self.counter)}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_name = self.env.setdefault(node.name, self.fresh())
        node.name = new_name
        
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        
        self.generic_visit(node.args)
        
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        new_name = self.env.setdefault(node.name, self.fresh())
        node.name = new_name
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        new = self.env.setdefault(node.id, self.fresh())
        return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        new = self.env.setdefault(node.arg, self.fresh())
        node.arg = new
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node.value)
        new_attr = self.env.setdefault(node.attr, self.fresh())
        node.attr = new_attr
        return node

    def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        return None

def normalize(code: str) -> ast.AST:
    tree = ast.parse(code)
    if (tree.body and isinstance(tree.body[0], ast.Expr) and
        isinstance(tree.body[0].value, ast.Constant) and
        isinstance(tree.body[0].value.value, str)):
        tree.body.pop(0)
    renamer = AlphaRenamer()
    return renamer.visit(tree)

def find_divergence(tree1: ast.AST, tree2: ast.AST
                   ) -> Optional[Tuple[int, int]]:
    for node1, node2 in zip(ast.walk(tree1), ast.walk(tree2)):
        if type(node1) is not type(node2):
            return (getattr(node1, 'lineno', None),
                    getattr(node2, 'lineno', None))

        if isinstance(node1, ast.BinOp):
            if type(node1.op) is not type(node2.op):
                return (node1.lineno, node2.lineno)

        if isinstance(node1, ast.Constant):
            if node1.value != node2.value:
                return (node1.lineno, node2.lineno)

        if isinstance(node1, ast.Compare):
            ops1 = [type(o) for o in node1.ops]
            ops2 = [type(o) for o in node2.ops]
            if ops1 != ops2:
                return (node1.lineno, node2.lineno)
    return None

def normalize_line_for_comparison(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    
    import re
    
    normalized = re.sub(r'["\'].*?["\']', 'STR', stripped)
    
    normalized = re.sub(r'\b\d+\.?\d*\b', 'NUM', normalized)
    
    keywords = {
        'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 
        'finally', 'with', 'return', 'yield', 'import', 'from', 'as', 'pass',
        'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'None', 'True', 
        'False', 'lambda', 'global', 'nonlocal', 'assert', 'del', 'raise'
    }
    
    tokens = re.findall(r'\b\w+\b|[^\w\s]', normalized)
    normalized_tokens = []
    
    for token in tokens:
        if token.isalpha() and token not in keywords:
            normalized_tokens.append('ID')
        else:
            normalized_tokens.append(token)
    
    return ''.join(normalized_tokens)

def get_common_indent(line1: str, line2: str) -> int:
    indent1 = len(line1) - len(line1.lstrip())
    indent2 = len(line2) - len(line2.lstrip())
    return min(indent1, indent2)

def is_divergence_at_line_start(lineA: str, lineB: str) -> bool:
    if lineA == "<Out of bounds>" or lineB == "<Out of bounds>":
        return False
    
    common_indent = get_common_indent(lineA, lineB)
    lineA_dedented = lineA[common_indent:] if common_indent < len(lineA) else lineA.lstrip()
    lineB_dedented = lineB[common_indent:] if common_indent < len(lineB) else lineB.lstrip()
    
    lineA_first_char = lineA_dedented[0] if lineA_dedented else ''
    lineB_first_char = lineB_dedented[0] if lineB_dedented else ''
    
    lineA_starts_with_space = lineA_first_char in [' ', '\t']
    lineB_starts_with_space = lineB_first_char in [' ', '\t']
    
    if lineA_starts_with_space != lineB_starts_with_space:
        return True
    
    import re
    
    lineA_tokens = re.findall(r'\S+', lineA_dedented)
    lineB_tokens = re.findall(r'\S+', lineB_dedented)
    
    if not lineA_tokens or not lineB_tokens:
        return lineA_dedented.strip() != lineB_dedented.strip()
    
    normalized_A = normalize_line_for_comparison(lineA_dedented)
    normalized_B = normalize_line_for_comparison(lineB_dedented)
    
    if normalized_A != normalized_B:
        first_token_A = lineA_tokens[0]
        first_token_B = lineB_tokens[0]
        
        structural_tokens = {
            'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'finally', 'with', 'return', 'yield', 'import', 'from', 'pass',
            'break', 'continue', 'assert', 'del', 'raise', '=', '+=', '-=', '*=', '/='
        }
        
        if (first_token_A in structural_tokens or first_token_B in structural_tokens):
            return first_token_A != first_token_B
        
        return normalized_A != normalized_B
    
    return False

def extract_all_code_pairs_with_filter(logfile: str):
    with open(logfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    failed_ids = set()
    for line in reversed(lines[-20:]):
        if "failed task ids" in line:
            match = re.search(r"failed task ids:\s*(\[.*\])", line)
            if match:
                try:
                    failed_ids = set(ast.literal_eval(match.group(1)))
                except Exception as e:
                    print("Error parsing failed task list:", e)
            break

    results = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if re.search(r'##### HumanEval/\d+ #####', line):
            match = re.search(r'HumanEval/\d+', line)
            task_id = match.group(0) if match else None

            if task_id not in failed_ids:
                i += 1
                continue

            # Extract prompt + solution + prediction
            prompt, solution, prediction = [], [], []
            state = 'search_prompt'

            while i < len(lines):
                line = lines[i]

                if state == 'search_prompt':
                    if re.search(r'##### PROMPT #####', line):
                        state = 'in_prompt'
                        prompt = []
                elif state == 'in_prompt':
                    if re.search(r'##### SOLUTION #####', line):
                        state = 'in_solution'
                        solution = []
                    else:
                        prompt.append(line)
                elif state == 'in_solution':
                    if re.search(r' - STEP:\d+', line):
                        state = 'search_prediction'
                    else:
                        solution.append(line)
                elif state == 'search_prediction':
                    if re.search(r'##### PREDICTION-1 #####', line):
                        state = 'in_prediction'
                        prediction = []
                elif state == 'in_prediction':
                    if line.strip() == '':
                        pass
                    elif not line.startswith((' ', '\t')):
                        break
                    else:
                        prediction.append(line)

                i += 1

            if prompt and solution and prediction:
                codeA = ''.join(prompt + solution)
                codeB = ''.join(prompt + prediction)
                results.append((task_id, codeA, codeB))
        else:
            i += 1

    return results



if __name__ == "__main__":
    path = "data/deepseek-6.7b_sample.log"
    code_pairs = extract_all_code_pairs_with_filter(path)
    print(f"Extracted {len(code_pairs)} pairs of code segments")

    output_path = "data/divergence_report.txt"
    
    total_divergences = 0
    divergences_at_line_start = 0

    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, (task_id, codeA, codeB) in enumerate(code_pairs):
            fout.write(f"\n=== {task_id} ===\n")

            try:
                t1 = normalize(codeA)
                t2 = normalize(codeB)
                diff = find_divergence(t1, t2)

                if diff and diff[0] is not None and diff[1] is not None:
                    total_divergences += 1
                    
                    codeA_lines = codeA.splitlines()
                    codeB_lines = codeB.splitlines()
                    
                    lineA = codeA_lines[diff[0]-1] if 0 < diff[0] <= len(codeA_lines) else "<Out of bounds>"
                    lineB = codeB_lines[diff[1]-1] if 0 < diff[1] <= len(codeB_lines) else "<Out of bounds>"

                    fout.write(f"The divergence occurs at the line: {diff}\n")
                    fout.write(f"Code A divergence line content: {lineA}\n")
                    fout.write(f"Code B divergence line content: {lineB}\n")
                    
                    # Check if divergence is at line start
                    is_at_start = is_divergence_at_line_start(lineA, lineB)
                    
                    if is_at_start:
                        divergences_at_line_start += 1
                        fout.write("Divergence occurs at line start: YES\n")
                    else:
                        fout.write("Divergence occurs at line start: NO\n")
                else:
                    fout.write("No divergence found\n")

            except Exception as e:
                fout.write("Analysis failed, error message: {}\n".format(str(e)))

        fout.write(f"\n\n=== SUMMARY ===\n")
        fout.write(f"Total divergences analyzed: {total_divergences}\n")
        fout.write(f"Divergences at line start: {divergences_at_line_start}\n")
        
        if total_divergences > 0:
            percentage = (divergences_at_line_start / total_divergences) * 100
            fout.write(f"Percentage of divergences at line start: {percentage:.2f}%\n")
        else:
            fout.write("No divergences found to calculate percentage\n")

    print(f"\nSUMMARY:")
    print(f"Total divergences analyzed: {total_divergences}")
    print(f"Divergences at line start: {divergences_at_line_start}")
    
    if total_divergences > 0:
        percentage = (divergences_at_line_start / total_divergences) * 100
        print(f"Percentage of divergences at line start: {percentage:.2f}%")
    else:
        print("No divergences found to calculate percentage")