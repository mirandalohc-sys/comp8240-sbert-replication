# COMP8240 Project: Replicating Sentence-BERT

Replication study of Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP-IJCNLP.

**Author:** Hoi Ching Lo (45759308), Macquarie University

## Status

- Assessment 1 (project proposal): submitted.
- Assessment 3 (final report): analysis code, LLM-judge materials and results added October 2026.

## Main results

| Evaluation | Measure | Result |
|---|---|---|
| STS Benchmark test (1,379 pairs), published | Spearman ρ × 100 | 77.03 |
| STS Benchmark test (1,379 pairs), this run | Spearman ρ × 100 | 76.99 |
| BIOSSES (100 pairs) | Spearman ρ × 100 | 51.99 |
| PAWS-Wiki test (8,000 pairs) | ROC-AUC | 0.623 |
| Pilot (10 pairs): SBERT vs human mean | Spearman ρ | 0.203 |
| Pilot (10 pairs): LLM judge vs human mean | Spearman ρ | 0.895 |

## Repository contents

- `code/analyse_study.py`: STS Benchmark, BIOSSES and PAWS-Wiki evaluation, baselines and pilot analysis
- `code/compare_biosses.py`: exploratory BIOSSES baseline comparison
- `code/analyse_judge.py`: LLM-as-judge analysis
- `judge_pack/`: the four judge prompts, protocol manifest and run log
- `judge_responses/`: the four collected ChatGPT responses (GPT-5.6 Sol High, 22 October 2026, memory disabled)
- `results/judge_evaluation.json`: judge comparison results
- `data/pilot_pairs_unrated.csv` and `data/annotation_guidelines.txt`: the ten pilot pairs and the guide given to raters
- `recorded_revisions.json`, `requirements.txt`: pinned model/data revisions and package versions
- `verify_sbert.py`, `sbert_verification.ipynb`, `Copy_of_sbert_verification.ipynb`: earlier verification work

## Reproducing the analysis

Python 3.12.14 on CPU was used. Full instructions are in `README.txt`.

    python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    python code/analyse_study.py --benchmarks-only --include-paws --output-dir run

The pilot and judge analyses also need the three private rating files, which are not public:

    python code/analyse_study.py --ratings-dir private_inputs --include-paws --output-dir run
    python code/analyse_judge.py --responses judge_responses --study-dir run

## Data and privacy

Benchmark data and model weights are downloaded from their original Hugging Face sources at the revisions pinned in `recorded_revisions.json`. Individual human ratings are kept private; only the unrated pilot pairs and aggregate results are published.

## Reference

Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 3982–3992.
