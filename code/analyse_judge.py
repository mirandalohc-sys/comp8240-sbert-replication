"""Analyse four independently collected judge runs without manufacturing scores.

Usage: python code/analyse_judge.py --responses judge_responses --study-dir run
Each response is a JSON array containing pair_id, llm_score and reason.
Preserve raw ChatGPT responses separately, including failed attempts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyse_study import correlation_ci, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()
    pilot = pd.read_csv(args.study_dir / 'private/PILOT_SBERT_NLI_predictions.csv').sort_values('pair_id')
    ids = pilot.pair_id.tolist()
    human, sbert = pilot.human_mean.to_numpy(), pilot.cosine.to_numpy()
    scores, details = {}, []
    for run in ['A_original', 'A_reversed', 'B_original', 'B_reversed']:
        path = args.responses / f'{run}.json'
        # JSON parsing is strict: formatting repairs must be made and logged explicitly.
        response = json.loads(path.read_text())
        if not isinstance(response, list) or len(response) != len(ids):
            raise ValueError(f'{run}: expected one object per item')
        frame = pd.DataFrame(response)
        if not {'pair_id', 'llm_score', 'reason'} <= set(frame.columns):
            raise ValueError(f'{run}: missing required columns')
        if not frame.pair_id.is_unique or set(frame.pair_id) != set(ids):
            raise ValueError(f'{run}: pair identifiers differ from pilot')
        if not all(type(x) is int and 0 <= x <= 5 for x in frame.llm_score.tolist()):
            raise ValueError(f'{run}: every score must be an integer 0-5')
        frame = frame.set_index('pair_id').loc[ids]
        predicted = frame.llm_score.to_numpy(dtype=float)
        scores[run] = predicted
        row = {'run_id': run, 'response_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'items': len(ids), 'mean_absolute_difference_from_human_mean': float(np.abs(predicted-human).mean())}
        if np.unique(predicted).size < 2:
            row.update({'spearman_llm_human': None, 'spearman_llm_sbert': None,
                        'reason_undefined': 'The judge assigned a constant score.'})
        else:
            result = correlation_ci(human, predicted, pilot=True)
            row.update({'spearman_llm_human': result['spearman'],
                        'spearman_llm_human_ci95': [x/100 for x in result['spearman_ci95_x100']],
                        'bootstrap_valid': result['resamples_valid'],
                        'spearman_llm_sbert': float(spearmanr(predicted, sbert).statistic)})
        details.append(row)
    changes = []
    for first, second, factor in [('A_original','A_reversed','orientation under A'),
                                  ('B_original','B_reversed','orientation under B'),
                                  ('A_original','B_original','prompt under original orientation'),
                                  ('A_reversed','B_reversed','prompt under reversed orientation')]:
        delta=scores[second]-scores[first]
        changes.append({'factor':factor,'first':first,'second':second,
                        'items_changed':int(np.count_nonzero(delta)),
                        'mean_absolute_score_change':float(np.abs(delta).mean()),
                        'maximum_absolute_score_change':float(np.abs(delta).max())})
    merged=pilot[['pair_id','sentence1','sentence2','human_mean','cosine']].copy()
    for run, values in scores.items():
        merged[run]=values.astype(int)
    merged.to_csv(args.study_dir/'private/judge_comparison.csv',index=False)
    result={'status':'collected responses analysed; consult run log for model and session details',
            'runs':details,'sensitivity':changes,
            'sbert_human_spearman':float(spearmanr(human,sbert).statistic),
            'interpretation':'Four conditions are repeated system judgments, not independent human raters. Absolute score differences assume numerical distances on the 0-5 scale.'}
    write_json(args.study_dir/'results/judge_evaluation.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
