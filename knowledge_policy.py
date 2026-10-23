"""Capacity-constrained policy ablation: training-only heuristic thresholds."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from claims_ai import load

def run(data, out):
    X, y = load(data)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=71)
    records = []
    for i, (a, b) in enumerate(cv.split(X, y)):
        train, test = X.iloc[a], X.iloc[b]
        yt, ye = y.iloc[a], y.iloc[b].to_numpy()
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=71))
        model.fit(train, yt)
        p = model.predict_proba(test)[:, 1]
        rulecols = ["claims_per_patient", "mean_reimbursement", "inpatient_fraction"]
        cutoffs = {c: train[c].quantile(0.9) for c in rulecols}
        flag = np.column_stack([(test[c] > cutoffs[c]).to_numpy() for c in rulecols]).any(axis=1)
        order = np.argsort(-p, kind="stable")
        rule_inds = np.flatnonzero(flag)
        rule_rank = rule_inds[np.argsort(-p[rule_inds], kind="stable")]
        for budget in [0.05, 0.10, 0.20]:
            n = int(np.ceil(len(ye) * budget))
            model_review = order[:n]
            # Quotas are predeclared; no labels from evaluation fold affect selection.
            for quota in [0.0, 0.10, 0.25, 0.50]:
                k = min(n, int(np.floor(n * quota)))
                outside_top = rule_rank[~np.isin(rule_rank, model_review)]
                selected_rule = outside_top[:k]
                keep_model = order[~np.isin(order, selected_rule)][:n - len(selected_rule)]
                selected = np.concatenate([keep_model, selected_rule])
                assert len(selected) == n and len(np.unique(selected)) == n
                records.append({
                    "repeat": i//5 + 1, "fold": i%5 + 1, "budget": budget,
                    "knowledge_quota": quota, "reviewed": n,
                    "rule_replacements": len(selected_rule),
                    "rule_replacement_positive_count": int(ye[selected_rule].sum()),
                    "positive_labels_captured": int(ye[selected].sum()),
                    "precision": float(ye[selected].mean()),
                    "recall": float(ye[selected].sum() / ye.sum())
                })
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(dest / "policy_quota_ablation.csv", index=False)
    print(frame.groupby(["budget", "knowledge_quota"])[["precision", "recall", "rule_replacements"]].mean().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run(args.data, args.out)
