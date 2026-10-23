# Reproducible healthcare claims fraud-risk screening benchmark

This repository evaluates a healthcare provider-level fraud-risk screening benchmark using real claims data from [Mendeley Data](https://doi.org/10.17632/gsn2hyty37.1). It compares logistic regression with heuristic indicator augmentation and also benchmarks a random forest and a standalone heuristic score.

**Data:** Download the four `Train*.csv` files from the cited dataset and place them in a local folder. The dataset is not redistributed here.

**Install:** `pip install -r requirements.txt`

**Held-out evaluation (plus 5 × 5 repeated validation):**
```bash
python claims_ai.py --data /path/to/csv-folder --out results
```

**Four-model repeated validation:**
```bash
python extra_validation.py --data /path/to/csv-folder --out results
```

**Measured cross-validation results (mean AP):** logistic regression 0.690205; logistic regression plus heuristic knowledge feature 0.688519; random forest 0.690788; rule score only 0.149999. These results do **not** show a consistent advantage from augmenting logistic regression with the heuristic feature.

The provider-level label `PotentialFraud` is not adjudicated evidence of actual fraud. The rules are *expert-inspired heuristics*, not expert-validated knowledge; repeated cross-validation folds are dependent. This is a reproducibility-oriented experiment, **not a validated production fraud-detection system**.

Two figure previews are included as SVG vector files in `results/`: `figure1_repeated_validation.svg` (all 25 observed paired AP differences) and `figure2_precision_recall.svg` (sampled points from the measured held-out precision–recall curves). Running the scripts creates full-resolution PDF figures: `figure1_repeated_validation.pdf` and `figure2_precision_recall.pdf`. The SVG curve preview samples display points; use the regenerated PDF for the full curve. The dataset is required to regenerate outputs and is intentionally not redistributed.
## New: capacity-aware decision audit

Run `python decision_audit.py --data /path/to/csv-folder --out results`. This evaluates **3 × 5 provider-disjoint stratified folds** and reports precision, recall, and lift at fixed **5%, 10%, and 20% review capacities**, plus transparent disagreement with training-derived heuristic thresholds. The summary is in [results/decision_audit_summary.json](results/decision_audit_summary.json).

At 10% hypothetical review capacity, logistic regression captured on average **64.1% of positive proxy labels** (precision **59.5%**). The classifier/heuristic disagreement rate was **21.0%**. These are retrospective, correlated-resampling estimates and **not evidence of recovered fraud, validated knowledge or generalizable performance**. External, temporal and subject-matter expert validation remain outstanding.
