"""Replicate the unsupervised STS rows of Reimers and Gurevych (2019), Table 1.

The source paper reports Spearman rank correlation (x100) between the cosine
similarity of sentence embeddings and gold scores on STS12-STS16, the STS
Benchmark test split and SICK-R. This script reruns that evaluation for four of
the paper's rows with released models only. No training, fine-tuning or
threshold selection is performed.

Usage, from the repository root:
    python code/replicate_table1.py --output-dir run_table1
    python code/replicate_table1.py --output-dir run_table1 --models SBERT_NLI_BASE AVG_GLOVE

A GPU is optional; on CPU the large model takes much longer.
Outputs in <output-dir>/results/:
    table1_replication.json   every score, with the paper value and the difference
    table1_replication.csv    the same as a table
    table1_rows.tex           LaTeX rows for the report
    table1_environment.json   packages, device, timings, pinned revisions
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

COLUMNS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSb", "SICK-R"]

# Reimers and Gurevych (2019), Table 1, Spearman x 100.
PAPER = {
    "AVG_GLOVE": [55.14, 70.66, 59.73, 68.25, 63.66, 58.02, 53.76],
    "AVG_BERT": [38.78, 57.98, 57.98, 63.15, 61.06, 46.35, 58.40],
    "SBERT_NLI_BASE": [70.97, 76.53, 73.19, 79.09, 74.30, 77.03, 72.91],
    "SBERT_NLI_LARGE": [72.27, 78.46, 74.90, 80.99, 76.25, 79.23, 73.75],
}
# The paper's own Avg. column. For SBERT-NLI-base it reports 74.89, although
# the mean of the seven printed values is 74.86; the printed value is kept.
PAPER_AVG = {"AVG_GLOVE": 61.32, "AVG_BERT": 54.81, "SBERT_NLI_BASE": 74.89, "SBERT_NLI_LARGE": 76.55}
LABELS = {
    "AVG_GLOVE": "Avg. GloVe embeddings",
    "AVG_BERT": "Avg. BERT embeddings",
    "SBERT_NLI_BASE": "SBERT-NLI-base",
    "SBERT_NLI_LARGE": "SBERT-NLI-large",
}

# Pinned revisions (checked 22 October 2026).
MODELS = {
    "AVG_GLOVE": ("sentence-transformers/average_word_embeddings_glove.6B.300d",
                  "e5e8fec6971be8960cfaa853a77a6ddc62a265d7", "sentence_transformer"),
    "AVG_BERT": ("google-bert/bert-base-uncased",
                 "86b5e0934494bd15c9632b12f734a8a67f723594", "mean_pooling"),
    "SBERT_NLI_BASE": ("sentence-transformers/bert-base-nli-mean-tokens",
                       "160a52b38a51ae87295ec3eabcf11755e5d27a8d", "sentence_transformer"),
    "SBERT_NLI_LARGE": ("sentence-transformers/bert-large-nli-mean-tokens",
                        "a85b0d10e16867eb768e411c66f79475505b65dd", "sentence_transformer"),
}
DATASETS = {
    "STS12": ("mteb/sts12-sts", "test", "3a51344839ec620dd3b753e0945bdf6c7e37af3e"),
    "STS13": ("mteb/sts13-sts", "test", "f475812ca75868066186343ebf60e01f95c75164"),
    "STS14": ("mteb/sts14-sts", "test", "8ab5e540b20b58d42fa6b944d3accee9df3fbaca"),
    "STS15": ("mteb/sts15-sts", "test", "ae8e1ecbd3b3bcf5c1c8fd798290033715d98b93"),
    "STS16": ("mteb/sts16-sts", "test", "d8e3be175a943eecfea09c96795087771996197c"),
    "STSb": ("sentence-transformers/stsb", "test", "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"),
    "SICK-R": ("mteb/sickr-sts", "test", "b93b4e08b707e9c9bc478e543495c3ced0b9d81e"),
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_pairs(load_dataset, repo: str, split: str, revision: str) -> pd.DataFrame:
    frame = load_dataset(repo, split=split, revision=revision).to_pandas()
    frame = frame[["sentence1", "sentence2", "score"]]
    if frame.isna().any().any():
        raise ValueError(f"{repo}: missing values")
    if not (frame.sentence1.str.strip().ne("").all() and frame.sentence2.str.strip().ne("").all()):
        raise ValueError(f"{repo}: blank sentence")
    return frame


def build_model(name: str, device: str):
    from sentence_transformers import SentenceTransformer, models

    repo, revision, kind = MODELS[name]
    if kind == "mean_pooling":
        # Same construction as the BERT_MEAN baseline in analyse_study.py.
        transformer = models.Transformer(repo, max_seq_length=128,
                                         model_args={"revision": revision},
                                         tokenizer_args={"revision": revision})
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[transformer, pooling], device=device)
    else:
        model = SentenceTransformer(repo, revision=revision, device=device)
    model.eval()
    return model


def cosine_scores(model, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    # Each distinct sentence is encoded once; every original pair is kept.
    unique = list(dict.fromkeys(frame.sentence1.tolist() + frame.sentence2.tolist()))
    vectors = model.encode(unique, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    lookup = dict(zip(unique, vectors))
    return np.array([float(np.dot(lookup[a], lookup[b]))
                     for a, b in zip(frame.sentence1, frame.sentence2)])


def latex_rows(rows: list[dict]) -> str:
    lines = []
    for name in [m for m in MODELS if any(r["model"] == m for r in rows)]:
        mine = {r["dataset"]: r["spearman_x100"] for r in rows if r["model"] == name}
        paper = PAPER[name]
        ours = [mine.get(c) for c in COLUMNS]
        paper_cells = [f"{v:.2f}" for v in paper] + [f"{PAPER_AVG[name]:.2f}"]
        our_cells = ["---" if v is None else f"{v:.2f}" for v in ours]
        our_cells.append(f"{np.mean(ours):.2f}" if None not in ours else "---")
        lines.append(f"{LABELS[name]} (paper) & " + " & ".join(paper_cells) + r" \\")
        lines.append(r"\quad this run & " + " & ".join(our_cells) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    np.random.seed(8240)
    torch.manual_seed(8240)
    results = args.output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)

    environment = {
        "python": platform.python_version(), "platform": platform.platform(), "device": device,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {p: version(p) for p in ["torch", "sentence-transformers", "transformers",
                                             "datasets", "scipy", "numpy", "pandas"]},
        "model_revisions": {k: {"repo": v[0], "revision": v[1]} for k, v in MODELS.items() if k in args.models},
        "dataset_revisions": {k: {"repo": v[0], "split": v[1], "revision": v[2]} for k, v in DATASETS.items()},
        "notes": ["Spearman is computed over all pairs of each dataset (pooled), not averaged over sub-collections.",
                  "The mteb STS12-STS16 and SICK-R copies store lower-cased, tokenised text; SICK-R has all 9,927 pairs.",
                  "The paper does not state which SICK-R split or which STS12-STS16 aggregation it used."],
        "timings_seconds": {},
    }
    write_json(results / "table1_environment.json", environment)

    data = {}
    for column in COLUMNS:
        repo, split, revision = DATASETS[column]
        data[column] = load_pairs(load_dataset, repo, split, revision)
        print("DATASET", column, repo, split, "pairs", len(data[column]), flush=True)

    rows = []
    for name in args.models:
        start = time.time()
        print("LOADING MODEL", name, MODELS[name][0], flush=True)
        model = build_model(name, device)
        for i, column in enumerate(COLUMNS):
            frame = data[column]
            predicted = cosine_scores(model, frame, args.batch_size)
            rho = float(spearmanr(frame.score.to_numpy(), predicted).statistic)
            row = {"model": name, "dataset": column, "pairs": len(frame),
                   "spearman_x100": rho * 100, "paper_x100": PAPER[name][i],
                   "difference_x100": rho * 100 - PAPER[name][i]}
            rows.append(row)
            print("RESULT", json.dumps(row), flush=True)
            write_json(results / "table1_replication.json", rows)
        environment["timings_seconds"][name] = round(time.time() - start, 1)
        write_json(results / "table1_environment.json", environment)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(results / "table1_replication.csv", index=False)
    (results / "table1_rows.tex").write_text(latex_rows(rows))
    summary = table.groupby("model").difference_x100.agg(
        mean_abs_difference=lambda d: float(np.abs(d).mean()),
        max_abs_difference=lambda d: float(np.abs(d).max()))
    environment["summary"] = summary.reset_index().to_dict(orient="records")
    environment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(results / "table1_environment.json", environment)
    print(table.pivot(index="model", columns="dataset", values="spearman_x100")[COLUMNS].round(2).to_string())
    print(summary.round(2).to_string())
    print("COMPLETE", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
