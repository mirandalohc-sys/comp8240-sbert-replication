SBERT CHECKPOINT RE-EVALUATION AND PILOT ANALYSIS

This folder is the code-release subset of the COMP8240 project. It contains no human scores or participant details. Copy its files into the project repository while retaining earlier notebooks and merging any existing .gitignore rules.

Python 3.12.14 was used. Create and activate a virtual environment, then install:
  python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r requirements.txt

Run the public benchmarks:
  python code/analyse_study.py --benchmarks-only --include-paws --output-dir run

This downloads pinned versions of STSB, BIOSSES, PAWS-Wiki, the original SBERT checkpoint and BERT-base. It applies no model training, then writes environment details, data audits, predictions and metrics under run/. Keep run/ out of version control unless you have checked the contents and relevant data permissions. Bootstrap resampling uses seed 8240.

To analyse the private pilot, the assessor needs the separately supplied pilot_rater_1.csv, pilot_rater_2.csv and pilot_rater_3.csv files:
  python code/analyse_study.py --ratings-dir private_inputs --include-paws --output-dir run

Only human agreement:
  python code/analyse_study.py --ratings-dir private_inputs --skip-models --output-dir run

Exploratory BIOSSES comparison after benchmark inference:
  python code/compare_biosses.py --study-dir run

The judge analysis script requires four actually collected JSON responses. It does not generate or substitute judge scores:
  python code/analyse_judge.py --responses judge_responses --study-dir run
The four collected responses (GPT-5.6 Sol High, 22 October 2026, memory disabled) are in judge_responses/; the run log is judge_pack/run_log.csv and the computed comparison is results/judge_evaluation.json.

Unrated pilot sentences (written by the author and checked with ChatGPT) are in data/pilot_pairs_unrated.csv; the annotation guide given to raters is data/annotation_guidelines.txt. Individual participant ratings are intentionally absent: public sharing was not discussed. The code and input schema support private verification without claiming that the human data are public.

STSB and BIOSSES use Spearman correlation; Pearson is supplementary. PAWS uses ROC-AUC and average precision. Pilot agreement uses ordinal Krippendorff alpha, checked by an independent implementation. Details and scope limitations are in the report. The available evidence reproduces one checkpoint result, not all training or experiments in the source paper.

Sources and attribution:
Reimers and Gurevych (2019): https://aclanthology.org/D19-1410/
SBERT: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
BERT: https://huggingface.co/google-bert/bert-base-uncased
STSB: https://huggingface.co/datasets/sentence-transformers/stsb
BIOSSES: https://huggingface.co/datasets/tabilab/biosses
PAWS: https://huggingface.co/datasets/google-research-datasets/paws
Ordinal-alpha implementation: https://github.com/pln-fing-udelar/fast-krippendorff

Retain upstream attribution and check source terms before redistributing datasets or model weights. The requirements file lists third-party packages; each retains its own licence. AI assistance was used to prepare the project code and documentation. The student is responsible for its review and submission.
