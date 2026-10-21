# COMP8240 Project: Replicating Sentence-BERT

Replication study of Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP-IJCNLP.

**Author:** Hoi Ching Lo (45759308), Macquarie University

## Status

- Assessment 1 (project proposal): submitted.
- Assessment 3 (final report): analysis code, full Table 1 replication, LLM-judge materials and results added October 2026.

## Main results

Replication of the paper's unsupervised STS table (Spearman ρ × 100, released models, no training):

| Model | | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R |
|---|---|---|---|---|---|---|---|---|
| SBERT-NLI-base | paper | 70.97 | 76.53 | 73.19 | 79.09 | 74.30 | 77.03 | 72.91 |
| | this run | 70.97 | 76.53 | 73.19 | 79.09 | 74.30 | 76.98 | 73.06 |
| SBERT-NLI-large | paper | 72.27 | 78.46 | 74.90 | 80.99 | 76.25 | 79.23 | 73.75 |
| | this run | 72.26 | 78.46 | 74.90 | 80.99 | 76.25 | 79.20 | 74.03 |

Averaged GloVe and averaged BERT baselines are in `results/table1_replication.csv`. The CPU run gives 76.99 for SBERT-NLI-base on STSb.

| Evaluation | Measure | Result |
|---|---|---|
| BIOSSES (100 pairs) | Spearman ρ × 100 | 51.99 |
| PAWS-Wiki test (8,000 pairs) | ROC-AUC | 0.623 |
| Pilot (10 pairs): SBERT vs human mean | Spearman ρ | 0.203 |
| Pilot (10 pairs): LLM judge vs human mean | Spearman ρ | 0.895 |

## Repository contents

- `code/replicate_table1.py`: the paper's full unsupervised STS table for four released models
- `code/analyse_study.py`: STS Benchmark, BIOSSES and PAWS-Wiki evaluation, baselines and pilot analysis
- `code/compare_biosses.py`: exploratory BIOSSES baseline comparison
- `code/analyse_judge.py`: LLM-as-judge analysis (needs the private rating files)
- `code/reproduce_pilot_public.py`: pilot and judge correlations from public files only
- `judge_pack/`: the four judge prompts, protocol manifest and run log
- `judge_responses/`: the four collected ChatGPT responses (GPT-5.6 Sol High, 22 October 2026, memory disabled)
- `results/`: Table 1 replication results and judge comparison
- `data/pilot_pairs_unrated.csv`, `data/pilot_human_means.csv`, `data/annotation_guidelines.txt`: the ten pilot pairs, their three-rater means and the guide given to raters
- `recorded_revisions.json`, `requirements.txt`: pinned model/data revisions and package versions
- `verify_sbert.py`, `sbert_verification.ipynb`, `Copy_of_sbert_verification.ipynb`: earlier verification work

## Reproducing the analysis

Full instructions are in `README.txt`.

    python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    python code/replicate_table1.py --output-dir run_table1
    python code/analyse_study.py --benchmarks-only --include-paws --output-dir run
    python code/reproduce_pilot_public.py

Inter-rater agreement (Krippendorff's alpha) needs the three private rating files, which are not public:

    python code/analyse_study.py --ratings-dir private_inputs --include-paws --output-dir run

## Data and privacy

Benchmark data and model weights are downloaded from their original Hugging Face sources at pinned revisions. Individual human ratings are kept private; only the unrated pilot pairs and the three-rater means (already shown in the report) are published.

## Reference

Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 3982–3992.
