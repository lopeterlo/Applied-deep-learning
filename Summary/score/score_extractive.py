import argparse
import json
import logging
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from rouge_score.rouge_scorer import RougeScorer

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
USE_STEMMER = False


def main(args):
    with open(args.predict_path) as f:
        predicts = [json.loads(line) for line in f]
        predicts = {p['id']: p for p in predicts}

    with open(args.answer_path) as f:
        answers = [json.loads(line) for line in f]
        answers = {a['id']: a for a in answers}

    if len(predicts) != len(answers):
        logging.warning(
            'Number of predicts ({}) and number of answers ({}) is different'
            .format(len(predicts), len(answers))
        )

    target = []
    prediction = []
    count = 0
    for id, p in predicts.items():
        a = answers.get(id, None)
        if a is None:
            raise Exception(f"Cannot find answer to Prediction ID: {id}")
        else:
            sent_bounds = {i: bound for i, bound in enumerate(a['sent_bounds'])}
            predict_sent = ''
            for sent_idx in p['predict_sentence_index']:
                start, end = sent_bounds.get(sent_idx, (0, 0))
                predict_sent += a['text'][start:end]
            if a['extractive_summary'] in p['predict_sentence_index']:
                count+=1
        target.append(a['summary'])
        prediction.append(predict_sent)        

    rouge_scorer = RougeScorer(ROUGE_TYPES, use_stemmer=USE_STEMMER)
    with Pool(cpu_count()) as pool:
        scores = pool.starmap(rouge_scorer.score,
                              [(t, p) for t, p in zip(target, prediction)])
    r1s = np.array([s['rouge1'].fmeasure for s in scores])
    r2s = np.array([s['rouge2'].fmeasure for s in scores])
    rls = np.array([s['rougeL'].fmeasure for s in scores])
    scores = {
        'mean': {
            'rouge-1': r1s.mean(),
            'rouge-2': r2s.mean(),
            'rouge-l': rls.mean()
        },
        'std': {
            'rouge-1': r1s.std(),
            'rouge-2': r2s.std(),
            'rouge-l': rls.std()
        },
    }
    print(json.dumps(scores, indent='    '))
    print(count)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the score of prediction."
    )
    parser.add_argument('predict_path', type=str,
                        help='')
    parser.add_argument('answer_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)