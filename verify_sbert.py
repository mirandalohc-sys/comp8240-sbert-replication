"""Reproduce the STS-benchmark evaluation of Reimers & Gurevych (2019),
"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (EMNLP).

Run (CPU is sufficient; ~4 minutes, plus a one-off ~438 MB model download):
    pip install -U sentence-transformers scipy datasets
    python3 verify_sbert.py

Published value (Table 1/2 of the paper): 77.03
Value obtained by this script: 76.99

Note: the sentence-transformers/stsb copy of the benchmark stores gold
scores normalised to 0-1 rather than the original 0-5. This does not
affect the result: Spearman's rho depends only on rank order.
"""
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, util

ds = load_dataset("sentence-transformers/stsb", split="test")
s1, s2 = list(ds["sentence1"]), list(ds["sentence2"])
gold = list(ds["score"])
print(f"{len(gold)} test pairs loaded (expected: 1379)")

model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
e1 = model.encode(s1, batch_size=32, show_progress_bar=True,
                  convert_to_tensor=True)
e2 = model.encode(s2, batch_size=32, show_progress_bar=True,
                  convert_to_tensor=True)
cos = util.cos_sim(e1, e2).diagonal().cpu().numpy()

res = spearmanr(gold, cos)
rho = (res.statistic if hasattr(res, "statistic") else res.correlation) * 100
print(f"Spearman rho x100 on STS-B test: {rho:.2f}   (published: 77.03)")
