import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import argparse
import os
import json

def load_and_preprocess_data(file_path, target_list):
    df = pd.read_parquet(file_path)
    print(f"\nTotal number of samples: {len(df)}")
    print(f"Number of samples with Rank > 50: {len(df[df['Rank'] > 50])}")
    
    df['y'] = (df['Rank'] == 1).astype(int)
    df = df.dropna(subset=target_list)
    
    print(f"Number of positive samples: {len(df[df['y'] == 1])}")
    print(f"Number of negative samples: {len(df[df['y'] == 0])}")
    
    for col in target_list:
        df[col] = np.where(np.isfinite(df[col]), df[col], np.nan)
    df = df.dropna(subset=target_list)
    
    return df

def split_data(df, target_list, test_size=0.2, random_state=42):
    X = df[target_list]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, target_list):
    train_data = pd.concat([X_train, y_train], axis=1)
    positive_class = train_data[train_data['y'] == 1]
    negative_class = train_data[train_data['y'] == 0]
    
    min_samples = min(len(positive_class), len(negative_class))
    train_positive = positive_class.sample(n=min_samples, random_state=42)
    train_negative = negative_class.sample(n=min_samples, random_state=42)
    
    train_data_balanced = pd.concat([train_positive, train_negative])
    X_train_balanced = train_data_balanced[target_list]
    y_train_balanced = train_data_balanced['y']
    
    model = LogisticRegression()
    model.fit(X_train_balanced, y_train_balanced)
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nEvaluation results with default threshold (0.5):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # ===== Find the best threshold (based on accuracy) =====
    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    print(f"\nBest threshold (to maximize Accuracy): {best_threshold:.2f}")
    print(f"Accuracy at best threshold: {best_accuracy:.4f}")

    # ===== Output complete evaluation results at the best threshold =====
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    precision_best = precision_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)
    auc_best = roc_auc_score(y_test, y_pred_proba)  # AUC is not threshold-dependent

    print(f"\nComplete evaluation results at best threshold:")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Precision: {precision_best:.4f}")
    print(f"Recall: {recall_best:.4f}")
    print(f"F1 Score: {f1_best:.4f}")
    print(f"AUC: {auc_best:.4f}")

    # ===== Derive the Entropy value from logistic regression =====
    if 'Entropy' in X_test.columns:
        coef_entropy = model.coef_[0][0]
        intercept = model.intercept_[0]
        point = best_threshold

        z_at_prob = np.log(point / (1 - point))
        entropy_at_prob = (z_at_prob - intercept) / coef_entropy
        print(f"\nLearned Entropy Threshold: {entropy_at_prob:.6f}")
        
        # ===== Save learned entropy threshold =====
        entropy_file = 'data/learned_thresholds.json'

        if os.path.exists(entropy_file):
            with open(entropy_file, 'r') as f:
                entropy_dict = json.load(f)
        else:
            entropy_dict = {}

        entropy_dict[model_name] = round(entropy_at_prob, 6)

        with open(entropy_file, 'w') as f:
            json.dump(entropy_dict, f, indent=4)

        print(f"Saved entropy threshold for {model_name} to '{entropy_file}'.")


def main(args):
    target_list = ['Entropy']
    df = load_and_preprocess_data(f'lr_train_data/{args.model}_statistics.parquet', target_list)
    
    X_train, X_test, y_train, y_test = split_data(df, target_list)
    
    model = train_model(X_train, y_train, target_list)
    
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    
    print('\nFormula: P(x=1) = 1 / (1 + exp(-(' + 
          f'{intercept:.4f}' + 
          ''.join([f' + {coef:.4f}*{name}' for coef, name in zip(coefficients, target_list)]) +
          ')))')
    
    evaluate_model(model, X_test, y_test, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True, type=str)
    args = parser.parse_args()
    main(args)