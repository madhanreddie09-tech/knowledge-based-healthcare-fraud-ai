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

## Policy ablation and manuscript (October 2026)

The paper uses a capacity-constrained AI/knowledge governance experiment. Run:

```bash
python knowledge_policy.py --data /path/to/csv-folder --out results
```

In 3 × 5 repeated stratified provider folds, **10% review capacity** captures **64.10%** of positive proxy labels under model-only selection. Reserving 10%, 25%, or 50% of review slots for heuristic-only flags changes recall to **63.24%, 59.49%, or 52.50%**, respectively. See `results/policy_quota_ablation_summary.json`. Code emits full 180-row fold-level `results/policy_quota_ablation.csv`.

These **negative intervention results are not evidence that heuristics improve fraud recovery**. The knowledge flags are statistical upper-decile conditions, not expert-validated rules. PotentialFraud labels are not verified legal adjudications. The reported folds overlap across repeats. This benchmark does not supply an external validation site, temporal holdout, prospective trial, calibrated cost model or journal acceptance guarantee.

## Rule-quota ablation (23 October 2026)

Run `python decision_policy_ablation.py --data /path/to/csv-folder --out results`. There are 540 measured policy-budget-fold observations, with three review budgets, eleven reserved-rule quotas, one rule-first baseline, and three repeats of five-fold provider-disjoint validation. At 10% review capacity, mean proxy-label recall was 64.10% with model-only ranking, unchanged with 40% minimum reservation, 58.76% with 80%, 43.28% with 100%, and 31.50% with rule-first ranking. See [selected measured results](results/decision_policy_ablation_key_results.csv). These are retrospective proxy-label results, not confirmed fraud detection. The six-page research manuscript with exactly two experimental figures and two results tables and the full 540-row output are in the companion submission ZIP. Independent external, temporal, operational and expert validation remain outstanding.

## Geographic transfer stress test (exploratory; not external validation)

Run `python geographic_stress.py --data /path/to/csv-folder --out results` to create a five-fold GroupKFold evaluation. Providers are grouped by their majority beneficiary State, with all 52 recorded state codes held out from training in one fold. The same underlying source dataset is used. Logistic regression mean proxy-label recall at 10% review capacity = **64.08%**; random forest = **64.77%**. [Summary](results/geographic_stress_summary.json). Beneficiaries can appear under multiple providers: there are **13,136–20,666 shared beneficiary IDs** between train and test provider groups across folds. Because of this overlap, the proxy outcome, and absence of another source dataset, **these results DO NOT establish external, temporally independent, or prospective generalization**. See the companion revised manuscript and external-validation protocol for details.

## External validation status (scientific integrity)

External validation **has not been established**. The five-fold grouped-state experiment reuses the same provider/beneficiary dataset and has substantial beneficiary overlap. Public CMS Medicare provider aggregates and HHS OIG LEIE exclusion files are independent data *candidates*, but they do not reproduce all of the frozen model's claim-/beneficiary-level input features or the `PotentialFraud` target; LEIE exclusion is not identical to adjudicated fraud. Independent validation therefore requires a second compatible provider/claims cohort, independently defined outcomes, a frozen-model evaluation, leakage and temporal audits, and qualified domain-expert review. Do not cite the source discovery as evidence that an external experiment was performed.
