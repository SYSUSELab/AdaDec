import os
import json
from typing import Optional, Sequence, Union, Dict
from tqdm import tqdm
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from adadec.generator import Generator


def _read_problems_bigcodebench(filename: str):
    problems = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                problem = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = problem.get("task_id") or problem.get("id") or f"line_{i}"
            problems[task_id] = problem
    return problems


def _train_logistic_and_find_threshold(df: pd.DataFrame,
                                       target_list: Sequence[str],
                                       test_size: float = 0.2,
                                       random_state: int = 42,
                                       lr_max_iter: int = 1000):
    df = df.copy()
    if 'Rank' not in df.columns:
        raise ValueError("input dataframe needs a 'Rank' column (like in your statistics.parquet).")
    df['y'] = (df['Rank'] == 1).astype(int)

    df = df.dropna(subset=target_list)
    for col in target_list:
        df[col] = np.where(np.isfinite(df[col]), df[col], np.nan)
    df = df.dropna(subset=target_list)

    if len(df) == 0:
        return None

    X = df[list(target_list)]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    train_data = pd.concat([X_train, y_train], axis=1)
    pos = train_data[train_data['y'] == 1]
    neg = train_data[train_data['y'] == 0]
    if len(pos) == 0 or len(neg) == 0:
        X_train_balanced = X_train
        y_train_balanced = y_train
    else:
        nmin = min(len(pos), len(neg))
        pos_s = pos.sample(n=nmin, random_state=random_state)
        neg_s = neg.sample(n=nmin, random_state=random_state)
        train_bal = pd.concat([pos_s, neg_s])
        X_train_balanced = train_bal[list(target_list)]
        y_train_balanced = train_bal['y']

    model = LogisticRegression(max_iter=lr_max_iter)
    model.fit(X_train_balanced, y_train_balanced)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.0
    best_accuracy = -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        y_pred_thresh = (y_pred_proba >= t).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(t)

    learned_entropy = None
    if 'Entropy' in X_test.columns:
        try:
            idx = list(X_train.columns).index('Entropy')
            coef_entropy = model.coef_[0][idx]
            intercept = float(model.intercept_[0])
            p = best_threshold
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)
            z = np.log(p / (1 - p))
            if coef_entropy == 0:
                learned_entropy = None
            else:
                entropy_at_prob = (z - intercept) / coef_entropy
                learned_entropy = float(entropy_at_prob)
        except ValueError:
            learned_entropy = None

    return {
        "model_obj": model,
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
        "learned_entropy": learned_entropy
    }


def prepare_adadec(model: Union[str, object],
                   tokenizer: Optional[object],
                   train_file: str = '',
                   generate_data_output_file: Optional[str] = None,
                   model_name: Optional[str] = None,
                   learned_thresholds_output_file: str = 'data/learned_thresholds.json',
                   target_list: Sequence[str] = ('Entropy',),
                   test_size: float = 0.2,
                   random_state: int = 42,
                   max_problems: Optional[int] = None,
                   verbose: bool = True) -> Dict:
    if os.path.exists(learned_thresholds_output_file):
        with open(learned_thresholds_output_file, 'r', encoding='utf-8') as f:
            _thresh_dict = json.load(f)
        if model_name in _thresh_dict:
            msg = (f"Threshold for model '{model_name}' already exists in "
                   f"'{learned_thresholds_output_file}'. Skipping re-training.")
            logging.warning(msg)
            return {
                "generate_data_file": generate_data_output_file,
                "learned_thresholds_file": learned_thresholds_output_file,
                "learned_entropy": _thresh_dict.get(model_name),
            }
    
    model_obj = model
    tokenizer_obj = tokenizer
    model_name_from_param = model_name or getattr(model_obj, "name", "model")

    if generate_data_output_file is None:
        generate_data_output_file = os.path.join('data', 'gt_guide_data', f"{model_name_from_param}_statistics.parquet")
        os.makedirs(os.path.dirname(generate_data_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(learned_thresholds_output_file), exist_ok=True) if os.path.dirname(learned_thresholds_output_file) else None

    problems = _read_problems_bigcodebench(train_file)
    items = list(problems.items())
    if max_problems:
        items = items[:max_problems]

    generator = Generator(model_obj, tokenizer_obj, model_name=model_name_from_param)
    if verbose:
        iterator = tqdm(items, desc="Generating gt-guide data", ascii=True)
    else:
        iterator = items

    for task_id, problem in iterator:
        prompt = problem.get("prompt")
        ground_truth = problem.get("canonical_solution")
        try:
            generator.generate_base_on_ground_truth(prompt=prompt,
                                                   ground_truth=ground_truth,
                                                   filename=generate_data_output_file)
        except Exception as e:
            if verbose:
                logging.warning(f"failed to generate for task {task_id}: {e}")
            continue

    if not os.path.exists(generate_data_output_file):
        raise FileNotFoundError(f"generate_data_output_file not found: {generate_data_output_file}")

    df = pd.read_parquet(generate_data_output_file)
    train_result = _train_logistic_and_find_threshold(df, target_list, test_size=test_size, random_state=random_state)

    learned_entropy_value = None

    if train_result is not None:
        learned_entropy_value = train_result.get("learned_entropy")

        if os.path.exists(learned_thresholds_output_file):
            with open(learned_thresholds_output_file, 'r', encoding='utf-8') as f:
                thresh_dict = json.load(f)
        else:
            thresh_dict = {}

        thresh_dict[model_name_from_param] = None if learned_entropy_value is None else round(float(learned_entropy_value), 6)
        
        os.makedirs(os.path.dirname(learned_thresholds_output_file), exist_ok=True) if os.path.dirname(learned_thresholds_output_file) else None
        with open(learned_thresholds_output_file, 'w', encoding='utf-8') as f:
            json.dump(thresh_dict, f, indent=4)

    return {
        "generate_data_file": generate_data_output_file,
        "learned_thresholds_file": learned_thresholds_output_file,
        "learned_entropy": learned_entropy_value,
    }
