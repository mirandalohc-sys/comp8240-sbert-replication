"""Exploratory paired bootstrap: token Jaccard minus SBERT on BIOSSES."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, spearmanr

from analyse_study import lexical_overlap, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    args=parser.parse_args()
    data=pd.read_csv(args.study_dir/'private/BIOSSES_SBERT_NLI_predictions.csv')
    gold=data.score.to_numpy()
    neural=data.cosine.to_numpy()
    lexical=np.array([lexical_overlap(a,b) for a,b in zip(data.sentence1,data.sentence2)])
    def difference(a,b,c):
        return spearmanr(a,b).statistic-spearmanr(a,c).statistic
    result=bootstrap((gold,lexical,neural),difference,paired=True,vectorized=False,
                     n_resamples=5000,method='BCa',random_state=np.random.default_rng(8240))
    output={'dataset':'BIOSSES','comparison':'Token Jaccard minus SBERT, Spearman',
            'difference_x100':float(difference(gold,lexical,neural)*100),
            'paired_bca_ci95_x100':[float(result.confidence_interval.low*100),float(result.confidence_interval.high*100)],
            'resamples':5000,'seed':8240,'note':'Exploratory baseline comparison; not pre-registered.'}
    write_json(args.study_dir/'results/biosses_baseline_difference.json',output)
    print(output)


if __name__=='__main__':
    main()
