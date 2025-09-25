# AdaDec: Adaptive Decoding for Code Generation

AdaDec is an adaptive decoding framework for LLM code generation. 
It selectively pauses decoding when uncertainty is high, reranks candidates, 
and improves accuracy with moderate overhead.

---

## 📦 Installation

```bash
pip install adadec
````

---

## ⚡ Quick Start

```python
from adadec import prepare_adadec, generate_adadec
prepare_adadec(model, tokenizer, "train.jsonl", "out.parquet", "mymodel", "thresholds.json")
result = generate_adadec(model, tokenizer, ["def add(a, b):"], "mymodel")
```

Note that train.jsonl requires "task_id", "prompt", and "canonical_solution".

stop_words.json example:
```json
[
  "\\n{4,}",
  "^\\S"
]
```



---

## 📖 License

MIT License















你这个 README.md 写得已经很不错了 👍，PyPI 上展示出来也会比较清晰。我帮你从 **PyPI 展示效果** 和 **开发者文档规范** 两个角度提几点小优化建议：

---

### ✅ 优点

* 有 **标题、简介、安装、使用示例**，这是 PyPI README 的核心部分。
* 英文部分简洁地说明了框架的动机和机制，读者一眼能理解。
* 提供了 **代码示例**，降低上手门槛。
* 附带了 **配置文件（stop\_words.json）示例**，很实用。

---

### 🔧 建议改进

1. **开头简介建议更简洁**
   目前第一段比较学术，可以再短一点，让 PyPI 页面上第一屏更直观，比如：

   ```markdown
   AdaDec is an adaptive decoding framework for LLM code generation. 
   It selectively pauses decoding when uncertainty is high, reranks candidates, 
   and improves accuracy with moderate overhead.
   ```

   然后再用一段详细说明工作机制（entropy trigger / lookahead 等）。

2. **中文内容与英文分离**
   PyPI 页面默认显示 README（通常是英文），如果 README 混杂中文和英文，部分用户可能看不懂。
   建议：

   * README.md 全部用英文
   * 如果需要中文说明，可以额外提供 `README_zh.md`，并在 README.md 里加链接：

     ```markdown
     [中文文档](https://github.com/yourname/adadec/blob/main/README_zh.md)
     ```

3. **加一个 Quick Start 小节**
   在 “Usage Example” 之前加个非常简短的 quick start，让用户三行跑起来，比如：

   ```python
   from adadec import prepare_adadec, generate_adadec
   prepare_adadec(model, tokenizer, "train.jsonl", "out.parquet", "mymodel", "thresholds.json")
   result = generate_adadec(model, tokenizer, ["def add(a, b):"], "mymodel")
   ```

   这样 PyPI 用户不用滚动太久就能复制粘贴试运行。

4. **License / Links**
   在 README 末尾加上：

   ```markdown
   ## 📖 License
   MIT License

   ## 🔗 Links
   - [Source Code](https://github.com/yourname/adadec)
   - [Issues](https://github.com/yourname/adadec/issues)
   ```

---

### ✨ 改进后的 README 结构示例

````markdown
# AdaDec: Adaptive Decoding for Code Generation

AdaDec is an adaptive decoding framework for LLM code generation. 
It selectively pauses decoding when uncertainty is high, reranks candidates, 
and improves accuracy with moderate overhead.

---

## 📦 Installation
```bash
pip install adadec
````

---

## ⚡ Quick Start

```python
from adadec import prepare_adadec, generate_adadec
prepare_adadec(model, tokenizer, "train.jsonl", "out.parquet", "mymodel", "thresholds.json")
result = generate_adadec(model, tokenizer, ["def add(a, b):"], "mymodel")
```

---

## 🚀 Usage Example

提供两个接口：`prepare_adadec()` 和 `generate_adadec()`。

```python
prepare_adadec(
    model=model,
    tokenizer=tokenizer,
    train_file="train_data.jsonl",
    generate_data_output_file="model_name_statistics.parquet",
    model_name=model_name,
    learned_thresholds_output_file="learned_thresholds.json"
)

prompts = ["def add(a, b):", "def factorial(n):"]

result = generate_adadec(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    model_name=model_name,
    beam_size=1,
    max_new_tokens=512,
    lambda_value=1.0,
    decoding_mode="AdaFixL",
    entropy_threshold="Learned",
    stop_words_file="stop_words.json"
)
```

训练数据要求有 `"task_id" "prompt" "canonical_solution"` 三个字段。

`stop_words.json` 示例：

```json
[
  "\\n{4,}",
  "^\\S"
]
```

---

## 📖 License

MIT License

## 🔗 Links

* [中文文档](https://github.com/yourname/adadec/blob/main/README_zh.md)
* [Source Code](https://github.com/yourname/adadec)
* [Issues](https://github.com/yourname/adadec/issues)

```

---

要不要我帮你把 **Usage Example 那段中文** 也翻译成英文？这样 PyPI 页面就完全是英文，显得更规范。
```
