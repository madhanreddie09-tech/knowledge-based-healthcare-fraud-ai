"""Four-model comparison using 5 × 5 repeated provider-level cross-validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from claims_ai import load, rules, thresholds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    X, y = load(args.data)
    results = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    for index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        xtrain, xtest = X.iloc[train_idx], X.iloc[test_idx]
        ytrain, ytest = y.iloc[train_idx], y.iloc[test_idx]
        imputer = SimpleImputer(strategy="median")
        train = pd.DataFrame(imputer.fit_transform(xtrain), columns=X.columns, index=xtrain.index)
        test = pd.DataFrame(imputer.transform(xtest), columns=X.columns, index=xtest.index)
        cutoff = thresholds(xtrain)
        configurations = [
            ("Logistic", make_pipeline(StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)), train, test),
            ("Logistic + knowledge", make_pipeline(StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                train.assign(knowledge_risk=rules(xtrain, cutoff)),
                test.assign(knowledge_risk=rules(xtest, cutoff))),
            ("Random forest", RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=2, random_state=42), train, test),
            ("Rule score only", None, None, None),
        ]
        for name, algorithm, train_features, test_features in configurations:
            if algorithm is None:
                predictions = rules(xtest, cutoff)
            else:
                algorithm.fit(train_features, ytrain)
                predictions = algorithm.predict_proba(test_features)[:, 1]
            results.append({
                "repeat": index // 5 + 1, "fold": index % 5 + 1, "model": name,
                "AP": average_precision_score(ytest, predictions),
                "AUC": roc_auc_score(ytest, predictions)})
        print("completed fold", index + 1, flush=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "model_comparison_25fold.csv", index=False)
    pivot = frame.pivot(index=["repeat", "fold"], columns="model", values="AP")
    diffs = pivot["Logistic + knowledge"] - pivot["Logistic"]
    summary = {
        "models": {name: {
            "mean_AP": float(group.AP.mean()),
            "sd_AP": float(group.AP.std()),
            "mean_AUC": float(group.AUC.mean())}
            for name, group in frame.groupby("model")},
        "paired_knowledge_delta_AP_mean": float(diffs.mean()),
        "paired_knowledge_positive_folds": int((diffs > 0).sum()),
        "repeats": 5, "folds": 5,
        "note": "Repeated folds are correlated; no independent-fold significance test."}
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.axhline(0, color="#555", lw=1, ls="--")
    ax.scatter(np.arange(1, 26), diffs,
               c=np.where(diffs >= 0, "#117c83", "#c65e40"), s=35)
    ax.axhline(diffs.mean(), color="#152c48", lw=1.5,
               label=f"Mean difference {diffs.mean():+.4f}")
    ax.set(xlabel="Paired provider-level validation fold (5 repeats × 5 folds)",
           ylabel="AP difference: augmented − baseline",
           title="Knowledge feature: unstable incremental utility")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure1_repeated_validation.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
