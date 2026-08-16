# COMP8240 Project: Replicating Sentence-BERT

Replication study of Reimers & Gurevych (2019), *Sentence-BERT:
Sentence Embeddings using Siamese BERT-Networks*, EMNLP-IJCNLP.

**Author:** Hoi Ching Lo (45759308), Macquarie University

## Status

Assessment 1 (project proposal) — submitted.

## Preliminary result

Re-running the authors' released `bert-base-nli-mean-tokens`
checkpoint on the STS benchmark test split (1,379 pairs):

| | Spearman rho x 100 |
|---|---|
| Published (Reimers & Gurevych, 2019) | 77.03 |
| This re-run | 76.99 |

CPU-only, under four minutes.

## Reproducing this

    pip install -U sentence-transformers scipy datasets
    python3 verify_sbert.py

## Planned work

- Full replication of released-checkpoint results
- Application to datasets the method has not been evaluated on
  (BIOSSES, SemEval-2015 PIT, Quora Question Pairs, PAWS-Wiki)
- A new hand-annotated sentence-pair set targeting negation,
  antonyms, numeric changes, and high-overlap traps
- An LLM-as-judge evaluation alongside the standard metrics

## Reference

Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence
Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 3982-3992.
