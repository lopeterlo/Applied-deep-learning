import argparse
import logging
import os
import json
import numpy as np
from rouge_score import rouge_scorer


def main(args):
    with open(args.predict_path) as f:
        predicts = [json.loads(line) for line in f]

    with open(args.answer_path) as f:
        answers = [json.loads(line) for line in f]

    for predict, answer in zip(predicts, answers):
        assert predict['id'] == answer['id']

    if len(predicts) != len(answers):
        logging.warning(
            'Number of predicts ({}) and number of answers ({}) is different'
            .format(len(predicts), len(answers))
        )
        predicts = predicts[:min(len(predicts), len(answers))]
        answers = answers[:min(len(predicts), len(answers))]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                      use_stemmer=False)
    scores = [
        scorer.score(
            a['summary'], p['predict']
        )
        for p, a in zip(predicts, answers)
    ]
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