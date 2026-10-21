"""Reproduce the pilot and LLM-judge results from public files only.

The full analysis (analyse_study.py, analyse_judge.py) needs the three private
rating files. This script needs only:
    data/pilot_pairs_unrated.csv   the ten sentence pairs
    data/pilot_human_means.csv     the three-rater mean per pair (as in Table 6)
    judge_responses/*.json         the four collected judge responses

Usage, from the repository root:
    python code/reproduce_pilot_public.py               # SBERT + judge results
    python code/reproduce_pilot_public.py --skip-sbert  # judge results only, no model download

Agreement statistics (Krippendorff's alpha) need individual ratings and are
therefore not reproduced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["A_original", "A_reversed", "B_original", "B_reversed"]
SBERT = ("sentence-transformers/bert-base-nli-mean-tokens", "160a52b38a51ae87295ec3eabcf11755e5d27a8d")


def rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def sbert_cosines(pairs: pd.DataFrame) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT[0], revision=SBERT[1], device="cpu")
    model.eval()
    a = model.encode(pairs.sentence1.tolist(), convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode(pairs.sentence2.tolist(), convert_to_numpy=True, normalize_embeddings=True)
    return (a * b).sum(axis=1)


def load_judge(responses: Path, ids: list[str]) -> pd.DataFrame:
    scores = {}
    for run in RUNS:
        frame = pd.DataFrame(json.loads((responses / f"{run}.json").read_text()))
        if set(frame.pair_id) != set(ids) or not frame.pair_id.is_unique:
            raise ValueError(f"{run}: pair identifiers differ from the pilot")
        if not all(type(x) is int and 0 <= x <= 5 for x in frame.llm_score):
            raise ValueError(f"{run}: every score must be an integer 0-5")
        scores[run] = frame.set_index("pair_id").llm_score.loc[ids]
    table = pd.DataFrame(scores)
    table["judge_mean"] = table[RUNS].mean(axis=1)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", type=Path, default=ROOT / "judge_responses")
    parser.add_argument("--output", type=Path, default=ROOT / "run_public" / "pilot_public_reproduction.json")
    parser.add_argument("--skip-sbert", action="store_true")
    args = parser.parse_args()

    pairs = pd.read_csv(ROOT / "data" / "pilot_pairs_unrated.csv").sort_values("pair_id")
    means = pd.read_csv(ROOT / "data" / "pilot_human_means.csv").set_index("pair_id")
    ids = pairs.pair_id.tolist()
    human = means.human_mean.loc[ids].to_numpy()
    judge = load_judge(args.responses, ids)

    result = {"pairs": len(ids), "reference": "three-rater mean per pair"}
    result["judge"] = {
        "spearman_per_run_vs_human": {run: rho(judge[run], human) for run in RUNS},
        "spearman_mean_judge_vs_human": rho(judge.judge_mean, human),
        "order_reversal_items_changed": int(((judge.A_original != judge.A_reversed).sum()
                                             + (judge.B_original != judge.B_reversed).sum())),
        "order_reversal_mean_abs_change": float(np.concatenate([
            (judge.A_original - judge.A_reversed).abs(), (judge.B_original - judge.B_reversed).abs()]).mean()),
        "P07_mean_judge_score": float(judge.judge_mean.loc["P07"]),
    }
    if not args.skip_sbert:
        cosine = sbert_cosines(pairs)
        loo = [rho(np.delete(human, k), np.delete(cosine, k)) for k in range(len(ids))]
        result["sbert"] = {
            "cosine": dict(zip(ids, map(float, cosine))),
            "spearman_vs_human": rho(cosine, human),
            "pearson_vs_human": float(pearsonr(cosine, human).statistic),
            "leave_one_pair_out_range": [min(loo), max(loo)],
        }
        result["judge"]["spearman_mean_judge_vs_sbert"] = rho(judge.judge_mean, cosine)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
