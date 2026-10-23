"""Provider-level healthcare fraud-risk screening benchmark, not clinical validation."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
SEED = 42

def load(root):
    root = Path(root)
    labels = pd.read_csv(root / "Train.csv")
    inpatient = pd.read_csv(root / "Train_Inpatientdata.csv")
    outpatient = pd.read_csv(root / "Train_Outpatientdata.csv")
    beneficiaries = pd.read_csv(root / "Train_Beneficiarydata.csv")
    assert labels.Provider.is_unique and beneficiaries.BeneID.is_unique
    claims = pd.concat([inpatient.assign(setting="inpatient"), outpatient.assign(setting="outpatient")], ignore_index=True)
    assert claims.ClaimID.is_unique
    cols = ["BeneID", "ChronicCond_Diabetes", "ChronicCond_Heartfailure"]
    claims = claims.merge(beneficiaries[cols], on="BeneID", how="left", validate="many_to_one")
    for col in ["InscClaimAmtReimbursed", "DeductibleAmtPaid", "ChronicCond_Diabetes", "ChronicCond_Heartfailure"]:
        claims[col] = pd.to_numeric(claims[col], errors="coerce")
    claims["chronic_count"] = claims[["ChronicCond_Diabetes", "ChronicCond_Heartfailure"]].eq(1).sum(axis=1)
    claims["inpatient"] = claims.setting.eq("inpatient").astype(int)
    grouped = claims.groupby("Provider").agg(
        claim_count=("ClaimID", "size"), unique_patients=("BeneID", "nunique"),
        mean_reimbursement=("InscClaimAmtReimbursed", "mean"),
        std_reimbursement=("InscClaimAmtReimbursed", "std"),
        total_reimbursement=("InscClaimAmtReimbursed", "sum"),
        mean_deductible=("DeductibleAmtPaid", "mean"),
        inpatient_fraction=("inpatient", "mean"),
        mean_chronic_count=("chronic_count", "mean"))
    grouped["claims_per_patient"] = grouped.claim_count / grouped.unique_patients.clip(lower=1)
    data = labels[["Provider", "PotentialFraud"]].merge(
        grouped, left_on="Provider", right_index=True, how="inner", validate="one_to_one")
    target = data.PotentialFraud.astype(str).str.lower().map({"yes": 1, "no": 0})
    assert target.notna().all() and target.nunique() == 2
    return data.drop(columns=["Provider", "PotentialFraud"]), target.astype(int)

def thresholds(train):
    return {"frequency": float(train.claims_per_patient.quantile(.9)),
            "amount": float(train.mean_reimbursement.quantile(.9)),
            "inpatient": float(train.inpatient_fraction.quantile(.9))}

def rules(frame, cutoffs):
    """Unvalidated heuristics, deliberately named rather than presented as expert rules."""
    return ((frame.claims_per_patient > cutoffs["frequency"]).astype(int)
            + (frame.mean_reimbursement > cutoffs["amount"]).astype(int)
            + (frame.inpatient_fraction > cutoffs["inpatient"]).astype(int)).to_numpy() / 3

def model():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])

def evaluate(y, p):
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p))}

def repeated_validation(X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    records = []
    for fold, (a, b) in enumerate(cv.split(X, y), 1):
        train, test = X.iloc[a], X.iloc[b]
        ytrain, ytest = y.iloc[a], y.iloc[b]
        cutoffs = thresholds(train)
        original = model()
        augmented = model()
        original.fit(train, ytrain)
        augmented.fit(train.assign(knowledge_risk=rules(train, cutoffs)), ytrain)
        p0 = original.predict_proba(test)[:, 1]
        p1 = augmented.predict_proba(test.assign(knowledge_risk=rules(test, cutoffs)))[:, 1]
        records.append({"fold": fold, "baseline_ap": average_precision_score(ytest, p0),
                        "augmented_ap": average_precision_score(ytest, p1),
                        "baseline_auc": roc_auc_score(ytest, p0),
                        "augmented_auc": roc_auc_score(ytest, p1)})
    frame = pd.DataFrame(records)
    frame["delta_ap"] = frame.augmented_ap - frame.baseline_ap
    frame["delta_auc"] = frame.augmented_auc - frame.baseline_auc
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    X, y = load(args.data)
    train, test, ytrain, ytest = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    cutoffs = thresholds(train)
    baseline, augmented = model(), model()
    baseline.fit(train, ytrain)
    augmented.fit(train.assign(knowledge_risk=rules(train, cutoffs)), ytrain)
    p0 = baseline.predict_proba(test)[:, 1]
    p1 = augmented.predict_proba(test.assign(knowledge_risk=rules(test, cutoffs)))[:, 1]
    cv = repeated_validation(X, y)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "repeated_cv.csv", index=False)
    rng = np.random.default_rng(SEED)
    labels = ytest.to_numpy()
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    differences = []
    for _ in range(2000):
        sample = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                 rng.choice(neg, len(neg), replace=True)])
        differences.append(average_precision_score(labels[sample], p1[sample])
                           - average_precision_score(labels[sample], p0[sample]))
    lower, upper = np.quantile(differences, [.025, .975])
    output = {
        "dataset": "Mendeley Data 10.17632/gsn2hyty37.1",
        "seed": SEED, "n_providers": len(y), "n_train": len(ytrain), "n_test": len(ytest),
        "positive_rate": float(y.mean()), "baseline": evaluate(ytest, p0),
        "knowledge_augmented": evaluate(ytest, p1),
        "delta_average_precision": float(average_precision_score(ytest, p1) - average_precision_score(ytest, p0)),
        "paired_stratified_bootstrap_delta_ap_ci95": [float(lower), float(upper)],
        "bootstrap_replicates": 2000,
        "repeated_cross_validation": {
            "repeats": 5, "folds": 5, "n_fold_results": len(cv),
            "mean_baseline_ap": float(cv.baseline_ap.mean()),
            "mean_augmented_ap": float(cv.augmented_ap.mean()),
            "mean_delta_ap": float(cv.delta_ap.mean()),
            "sd_delta_ap": float(cv.delta_ap.std()),
            "positive_delta_folds": int((cv.delta_ap > 0).sum()),
            "mean_delta_auc": float(cv.delta_auc.mean()),
            "caution": "Cross-validation fold results are dependent."},
        "training_only_thresholds": cutoffs,
        "note": "PotentialFraud is a dataset label, not adjudicated fraud."}
    (out / "metrics.json").write_text(json.dumps(output, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, predictions in [("Logistic regression", p0), ("Knowledge-augmented logistic", p1)]:
        precision, recall, _ = precision_recall_curve(ytest, predictions)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(ytest, predictions):.3f})")
    ax.axhline(ytest.mean(), ls="--", color="gray", label="Test prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Held-out provider fraud-risk screening")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure2_precision_recall.pdf")
    plt.close(fig)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
