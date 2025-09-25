# AdaDec: Adaptive Decoding for Code Generation

AdaDec is an adaptive decoding framework for LLM code generation. 
It selectively pauses decoding when uncertainty is high, reranks candidates, 
and improves accuracy with moderate overhead.


## Installation

```bash
pip install adadec
````


## Quick Start

```python
from adadec import prepare_adadec, generate_adadec
prepare_adadec(model, tokenizer, "train.jsonl", "out.parquet", "mymodel", "thresholds.json")
result = generate_adadec(model, tokenizer, ["def add(a, b):"], "mymodel")
```

Note that train.jsonl requires "task_id", "prompt", and "canonical_solution".

stop_words.json example:
```json
["\\n{4,}", "^\\S"]
```


## 📖 License

MIT License


## Links
* [Source Code](https://github.com/SYSUSELab/AdaDec)

