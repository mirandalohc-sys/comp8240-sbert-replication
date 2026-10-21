"""Evaluate the existing SBERT checkpoint and the completed ten-item pilot.

Raw human ratings and item-level outputs stay in the specified private directory.
No model is trained. External datasets and models are pinned by resolved revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
from scipy.stats import bootstrap, pearsonr, spearmanr


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manual_ordinal_alpha(ratings):
    """Independent implementation of ordinal coincidence-matrix alpha.

    Rows are raters, columns are items. This routine requires complete ratings.
    Formula: Krippendorff (2011), Computing Krippendorff's Alpha-Reliability.
    """
    ratings = np.asarray(ratings, dtype=int)
    m, _ = ratings.shape
    coincidences = np.zeros((6, 6), float)
    for column in ratings.T:
        counts = np.bincount(column, minlength=6)
        coincidences += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    counts = coincidences.sum(axis=0)
    expected = (np.outer(counts, counts) - np.diag(counts)) / (counts.sum() - 1)
    midpoint = np.cumsum(counts) - counts / 2
    distances = (midpoint[:, None] - midpoint[None, :]) ** 2
    return float(1 - (coincidences * distances).sum() / (expected * distances).sum())


def alpha_summary(ratings, n_resamples=5000, seed=8240):
    alpha = float(krippendorff.alpha(reliability_data=ratings, value_domain=np.arange(6), level_of_measurement="ordinal"))
    independent = manual_ordinal_alpha(ratings)
    if not np.isclose(alpha, independent, atol=1e-12):
        raise AssertionError("Ordinal alpha implementations disagree.")
    rng = np.random.default_rng(seed)
    samples = []
    invalid = 0
    for _ in range(n_resamples):
        sample = ratings[:, rng.integers(0, ratings.shape[1], ratings.shape[1])]
        try:
            value = krippendorff.alpha(reliability_data=sample, value_domain=np.arange(6), level_of_measurement="ordinal")
        except ValueError:
            invalid += 1
            continue
        if np.isfinite(value):
            samples.append(float(value))
        else:
            invalid += 1
    return {
        "ordinal_alpha": alpha,
        "independent_check": independent,
        "bootstrap_percentile_95_ci": np.quantile(samples, [.025, .975]).tolist(),
        "bootstrap_unit": "sentence pair; all three ratings retained together",
        "bootstrap_attempted": n_resamples,
        "bootstrap_valid": len(samples),
        "bootstrap_undefined": invalid,
        "seed": seed,
    }


def correlation_ci(human, predicted, pilot=False, seed=8240):
    human = np.asarray(human, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    point = float(spearmanr(human, predicted).statistic)
    if pilot:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(5000):
            index = rng.integers(0, len(human), len(human))
            if np.unique(human[index]).size < 2 or np.unique(predicted[index]).size < 2:
                continue
            values.append(float(spearmanr(human[index], predicted[index]).statistic))
        interval = np.quantile(values, [.025, .975])
        method = "paired percentile bootstrap; undefined constant resamples excluded"
        valid = len(values)
    else:
        result = bootstrap((human, predicted), lambda a, b: spearmanr(a, b).statistic,
                           paired=True, vectorized=False, n_resamples=5000,
                           batch=200, method="BCa", confidence_level=.95,
                           random_state=np.random.default_rng(seed))
        interval = np.array([result.confidence_interval.low, result.confidence_interval.high])
        method = "paired BCa bootstrap"
        valid = 5000
    if not np.isfinite(interval).all():
        raise ValueError("Undefined confidence interval.")
    return {"spearman": point, "spearman_x100": point * 100,
            "pearson": float(pearsonr(human, predicted).statistic),
            "spearman_ci95_x100": (interval * 100).tolist(),
            "bootstrap_method": method, "resamples_attempted": 5000,
            "resamples_valid": valid, "seed": seed}


def lexical_overlap(a, b):
    sa = set(re.findall(r"\w+", a.casefold()))
    sb = set(re.findall(r"\w+", b.casefold()))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratings-dir", type=Path)
    p.add_argument("--benchmarks-only", action="store_true")
    p.add_argument("--revisions-file", type=Path, default=Path(__file__).resolve().parents[1] / "recorded_revisions.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-paws", action="store_true")
    p.add_argument("--skip-models", action="store_true")
    args = p.parse_args()
    if not args.benchmarks_only and args.ratings_dir is None:
        p.error("Supply --ratings-dir or use --benchmarks-only.")
    if args.benchmarks_only and args.skip_models:
        p.error("--benchmarks-only cannot be combined with --skip-models.")
    output = args.output_dir
    private = output / "private"
    results = output / "results"
    private.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    start = time.time()
    merged = None
    if not args.benchmarks_only:
        frames = []
        input_hashes = {}
        for rater in range(1, 4):
            path = args.ratings_dir / f"pilot_rater_{rater}.csv"
            input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = pd.read_csv(path, keep_default_na=False).sort_values("pair_id").reset_index(drop=True)
            if len(frame) != 10 or not frame.pair_id.is_unique or set(frame.pair_id) != {f"P{k:02}" for k in range(1, 11)}:
                raise ValueError("Expected ten unique pair IDs in each rater file.")
            scores = pd.to_numeric(frame.human_score, errors="raise")
            if not ((scores >= 0) & (scores <= 5) & (scores % 1 == 0)).all():
                raise ValueError("Human scores must be whole numbers from zero to five.")
            frame.human_score = scores.astype(int)
            frames.append(frame)
        text_cols = ["pair_id", "sentence1", "sentence2"]
        for frame in frames[1:]:
            if not frame[text_cols].equals(frames[0][text_cols]):
                raise ValueError("Pair IDs or sentence text differ between files.")
        merged = frames[0][text_cols].copy()
        for i, frame in enumerate(frames, 1):
            merged[f"rater_{i}"] = frame.human_score
        values = merged[["rater_1", "rater_2", "rater_3"]].to_numpy().T
        merged["human_mean"] = values.mean(axis=0)
        merged["human_median"] = np.median(values, axis=0)
        merged["rating_range"] = values.max(axis=0) - values.min(axis=0)
        merged["category"] = ["negation", "negation", "antonymy", "antonymy", "numbers", "numbers", "word_overlap", "word_overlap", "Australian_English", "Australian_English"]
        merged.to_csv(private / "pilot_ratings_merged.csv", index=False)
        alpha = alpha_summary(values)
        alpha.update({"items": 10, "raters": 3, "ratings": 30,
                      "unanimous_items": int((merged.rating_range == 0).sum()),
                      "items_with_one_point_range": int((merged.rating_range == 1).sum()),
                      "score_counts": {str(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))},
                      "input_sha256": input_hashes,
                      "public_release_of_individual_ratings": "not authorised; public sharing was not discussed"})
        write_json(results / "pilot_agreement.json", alpha)
        print("PILOT AGREEMENT", json.dumps(alpha), flush=True)
    if args.skip_models:
        return

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer, models
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.set_num_threads(4)
    np.random.seed(8240)
    torch.manual_seed(8240)
    api = HfApi()
    revisions_path = results / "resolved_revisions.json"
    initial_revisions = json.loads(args.revisions_file.read_text()) if args.revisions_file.exists() else {}
    revisions = json.loads(revisions_path.read_text()) if revisions_path.exists() else initial_revisions
    if initial_revisions and revisions != initial_revisions:
        raise ValueError("Existing output revisions differ from the supplied revision manifest. Use a new output directory.")
    write_json(revisions_path, revisions)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "device": "cpu", "torch_threads": 4,
                   "started_utc": datetime.now(timezone.utc).isoformat(),
                   "packages": {name: version(name) for name in ["torch", "sentence-transformers", "transformers", "datasets", "huggingface-hub", "scipy", "numpy", "pandas", "scikit-learn", "krippendorff"]}}
    write_json(results / "environment.json", environment)

    def dataset(repo, split, config=None):
        key = f"dataset:{repo}"
        if key not in revisions:
            revisions[key] = api.dataset_info(repo).sha
            write_json(revisions_path, revisions)
        ds = load_dataset(repo, name=config, split=split, revision=revisions[key])
        df = ds.to_pandas()
        column = "label" if repo.endswith("paws") else "score"
        if df[["sentence1", "sentence2", column]].isna().any().any():
            raise ValueError("Missing evaluation data.")
        if any(not df[c].str.strip().ne("").all() for c in ["sentence1", "sentence2"]):
            raise ValueError("Blank evaluation sentence.")
        pairkeys = df.apply(lambda row: tuple(sorted([row.sentence1.casefold().strip(), row.sentence2.casefold().strip()])), axis=1)
        print("DATASET", repo, split, "pairs", len(df), "duplicate_unordered_pairs", int(pairkeys.duplicated().sum()), flush=True)
        return df, {"repo": repo, "revision": revisions[key], "split": split, "config": config,
                    "pairs": len(df), "label_min": float(df[column].min()), "label_max": float(df[column].max()),
                    "duplicate_unordered_pairs": int(pairkeys.duplicated().sum()), "missing_values": 0,
                    "label_column": column, "preprocessing": "original text retained; no case folding or deduplication for neural evaluation"}

    evaluation = {}
    metadata = {}
    for name, repo, split, config in [
        ("STSB", "sentence-transformers/stsb", "test", None),
        ("BIOSSES", "tabilab/biosses", "train", None),
    ]:
        evaluation[name], metadata[name] = dataset(repo, split, config)
    if merged is not None:
        evaluation["PILOT"] = merged.copy()
    if args.include_paws:
        evaluation["PAWS"] , metadata["PAWS"] = dataset("google-research-datasets/paws", "test", "labeled_final")
    write_json(results / "datasets.json", metadata)
    all_results = []
    modelspecs = [("SBERT_NLI", "sentence-transformers/bert-base-nli-mean-tokens"),
                  ("BERT_MEAN", "google-bert/bert-base-uncased")]
    for modelname, repo in modelspecs:
        key = f"model:{repo}"
        if key not in revisions:
            revisions[key] = api.model_info(repo).sha
            write_json(revisions_path, revisions)
        print("LOADING MODEL", modelname, revisions[key], flush=True)
        if modelname == "SBERT_NLI":
            model = SentenceTransformer(repo, revision=revisions[key], device="cpu")
        else:
            transformer = models.Transformer(repo, max_seq_length=128,
                model_args={"revision": revisions[key]}, tokenizer_args={"revision": revisions[key]})
            pool = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
            model = SentenceTransformer(modules=[transformer, pool], device="cpu")
        model.eval()
        for name, frame in evaluation.items():
            if modelname == "BERT_MEAN" and name == "PAWS":
                continue
            prediction_path = private / f"{name}_{modelname}_predictions.csv"
            print("EVALUATING", modelname, name, "max_seq_length", model.max_seq_length, flush=True)
            sentences = frame.sentence1.tolist() + frame.sentence2.tolist()
            unique = list(dict.fromkeys(sentences))
            embedding = model.encode(unique, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
            lookup = dict(zip(unique, embedding))
            predicted = np.array([np.dot(lookup[a], lookup[b]) for a, b in zip(frame.sentence1, frame.sentence2)])
            stored = frame.copy()
            stored["cosine"] = predicted
            stored.to_csv(prediction_path, index=False)
            row = {"dataset": name, "model": modelname, "pairs": len(frame), "max_seq_length": model.max_seq_length}
            if name == "PAWS":
                human = frame.label.to_numpy()
                row.update({"roc_auc": float(roc_auc_score(human, predicted)),
                            "average_precision": float(average_precision_score(human, predicted)),
                            "positive_prevalence": float(human.mean()),
                            "threshold_selected": False})
            else:
                human = frame.human_mean.to_numpy() if name == "PILOT" else frame.score.to_numpy()
                row.update(correlation_ci(human, predicted, pilot=name == "PILOT"))
                if name == "PILOT":
                    row["spearman_with_median"] = float(spearmanr(frame.human_median, predicted).statistic)
                    row["spearman_leave_one_pair_out"] = [float(spearmanr(np.delete(human, k), np.delete(predicted, k)).statistic) for k in range(len(human))]
            all_results.append(row)
            write_json(results / "evaluation.json", all_results)
            print("RESULT", json.dumps(row), flush=True)
        del model

    for name, frame in evaluation.items():
        predicted = np.array([lexical_overlap(a, b) for a, b in zip(frame.sentence1, frame.sentence2)])
        row = {"dataset": name, "model": "TOKEN_JACCARD", "pairs": len(frame),
               "note": "additional diagnostic baseline, not a replicated model from the SBERT paper"}
        if name == "PAWS":
            row.update({"roc_auc": float(roc_auc_score(frame.label, predicted)),
                        "average_precision": float(average_precision_score(frame.label, predicted))})
        else:
            gold = frame.human_mean if name == "PILOT" else frame.score
            row.update({"spearman": float(spearmanr(gold, predicted).statistic),
                        "spearman_x100": float(spearmanr(gold, predicted).statistic) * 100})
        all_results.append(row)
        print("RESULT", json.dumps(row), flush=True)
    write_json(results / "evaluation.json", all_results)
    environment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    environment["elapsed_seconds"] = time.time() - start
    write_json(results / "environment.json", environment)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
