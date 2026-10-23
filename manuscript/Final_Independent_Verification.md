# Independent verification — 23 October 2026

The corrected seven-page Word manuscript and PDF, two scientifically generated PDF figures, source scripts, and reproducibility CSVs are included in the companion course-submission ZIP (GitHub binary upload not established).

## Exact re-execution on supplied original dataset

- `python decision_policy_ablation.py --data <csv-folder> --out <output>`: generated 540 × 16 measurements; pandas `DataFrame.equals` exactly true against previously published CSV, all numeric max differences 0.
- `python geographic_stress.py --data <csv-folder> --out <output>`: generated 30 × 15 measurements; exact equality to previously published CSV.
- Four source modules pass Python compilation.
- Manuscript: 7 rendered pages, 2 research tables, 2 inline research figures, 24 references.

## Corrected citation

Reference [21], *XAI for Fraud Detection: An Attention-Based Ensemble of CNNs, GNNs, and Confidence-Driven Gating for Reliable Decision-Making* has **DOI 10.1109/TIFS.2025.3646480**, not the earlier erroneous identifier. Verified from [IEEE Xplore](https://ieeexplore.ieee.org/document/11305212/).

## Scope and limits

Year-specific Q1 category ranking has not been certified for each cited journal. This is not an independent external validation: both tests reuse the same underlying provider/beneficiary dataset, and the target is `PotentialFraud`, not adjudicated fraud. Domain-expert certification, independent cohort, funding/conflict declarations, and formal journal peer review remain outstanding. Do not label the paper journal-ready on these results alone.
