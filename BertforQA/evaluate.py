import argparse
import collections
import json
import os
from pathlib import Path
from pprint import pprint

import ckiptagger
import tensorflow as tf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path, help='Original data JSON file')
    parser.add_argument('prediction_path', type=Path, help='Model prediction JSON file')
    parser.add_argument('output_path', type=Path, help='Evaluation result save file')
    parser.add_argument('ckip_model_dir', type=Path, help='CKIP model directory')
    args = parser.parse_args()

    return vars(args)


def load_json(json_path):
    print(f'[*] Loading {json_path}...', end='', flush=True)
    with open(json_path) as f:
        result = json.load(f)
    print('done')

    return result


def save_json(data, json_path):
    print(f'[*] Saving to {json_path}...', end='', flush=True)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print('done')


def collect_answers(data):
    answers = {}
    for d in data['data']:
        for p in d['paragraphs']:
            for qa in p['qas']:
                answers[qa['id']] = {
                    'answerable': qa['answerable'],
                    'answers': [a['text'] for a in qa['answers']]
                }

    return answers


class Tokenizer:
    def __init__(self, model_dir):
        print(f'[*] Creating CKIP tokenizer from {model_dir}...', end='', flush=True)
        self._ws = ckiptagger.WS(model_dir)
        self._pos = ckiptagger.POS(model_dir)
        self._pos_punc_class_suffix = 'CATEGORY'
        print('done')

    def __call__(self, text, remove_punc=False):
        tokens = self._ws([text])[0]
        if not remove_punc:
            return tokens

        pos = self._pos([tokens])[0]
        tokens = [t for t, p in zip(tokens, pos)
                  if not p.endswith(self._pos_punc_class_suffix)]

        return tokens


def compute_em(ans, pred):
    def em(a, p):
        return int(''.join(a) == ''.join(p))

    return max([em(a, pred) for a in ans])


def compute_f1(ans, pred):
    def f1(a, p):
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        if len(a) == 0 or len(p) == 0:
            return int(''.join(a) == ''.join(p))

        common = collections.Counter(a) & collections.Counter(p)
        tp = sum(common.values())
        if tp == 0:
            return 0
        precision = tp / len(p)
        recall = tp / len(a)

        return (2 * precision * recall) / (precision + recall)

    return max([f1(a, pred) for a in ans])


def compute_metric(ans, pred, tokenizer):
    ans = [tokenizer(a, remove_punc=True) for a in ans]
    pred = tokenizer(pred, remove_punc=True)

    return {
        'em': compute_em(ans, pred),
        'f1': compute_f1(ans, pred)
    }


def compute_metrics(answers, predictions, tokenizer):
    metrics = []
    for id_ in tqdm(list(answers.keys()), desc='[*] Evaluating', dynamic_ncols=True):
        if id_ not in predictions:
            print(f'[!] Cannot find answer for id {id_} in model predictions')
            continue
        answerable = answers[id_]['answerable']
        prediction = predictions[id_]
        metric = compute_metric(answers[id_]['answers'], prediction, tokenizer)
        metrics.append({
            **metric,
            'answerable': answerable,
            'answerable_acc': int(answerable ^ (prediction == ''))
        })

    n_total = len(metrics)
    n_answerable = len([m for m in metrics if m['answerable']])
    n_unanswerable = n_total - n_answerable
    result = {
        'overall': {
            'count': n_total,
            'em': sum([m['em'] for m in metrics]) / n_total,
            'f1': sum([m['f1'] for m in metrics]) / n_total
        },
        'answerable': {
            'count': n_answerable,
            'em': sum([m['em'] for m in metrics if m['answerable']]) / n_answerable,
            'f1': sum([m['f1'] for m in metrics if m['answerable']]) / n_answerable
        },
        'unanswerable': {
            'count': n_unanswerable,
            'em': sum([m['em'] for m in metrics if not m['answerable']]) / n_unanswerable,
            'f1': sum([m['f1'] for m in metrics if not m['answerable']]) / n_unanswerable
        },
        'answerable accuracy': sum(m['answerable_acc'] for m in metrics) / n_total
    }

    return result


def main(data_path, prediction_path, output_path, ckip_model_dir):
    # Surpress TensorFlow and OpenMP messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["KMP_WARNINGS"] = "FALSE"

    # Set TensorFlow random seed
    tf.compat.v1.set_random_seed(19)

    print(f'[-] Original data file: {data_path}')
    print(f'[-] Model prediction file: {prediction_path}')
    print(f'[-] Evaluation output path: {output_path}\n')

    # Load gold answers
    data = load_json(data_path)
    answers = collect_answers(data)

    # Load model predictions
    predictions = load_json(prediction_path)

    # Create tokenizer
    tokenizer = Tokenizer(ckip_model_dir)

    # Compute metrics
    result = compute_metrics(answers, predictions, tokenizer)

    # Save evaluation result
    save_json(result, output_path)
    pprint(result)


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)