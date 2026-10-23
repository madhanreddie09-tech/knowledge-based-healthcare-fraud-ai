"""Knowledge-rule quota ablation; prespecified budget and provider-disjoint CV."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from claims_ai import load,rules
SEED=71

def run(dataset,out):
    x,y=load(dataset)
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=SEED)
    rows=[]
    for j,(a,b) in enumerate(cv.split(x,y)):
        tx,vx=x.iloc[a],x.iloc[b]
        ty,vy=y.iloc[a].to_numpy(),y.iloc[b].to_numpy()
        cut={"frequency":float(tx.claims_per_patient.quantile(.9)),
             "amount":float(tx.mean_reimbursement.quantile(.9)),
             "inpatient":float(tx.inpatient_fraction.quantile(.9))}
        rs=rules(vx,cut)
        flag=rs>0
        clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
            LogisticRegression(max_iter=1500,class_weight="balanced",random_state=SEED))
        clf.fit(tx,ty)
        p=clf.predict_proba(vx)[:,1]
        for budget in (.05,.10,.20):
            count=int(np.ceil(budget*len(vy)))
            ranked=np.argsort(-p,kind="stable")
            rule_ranked=np.lexsort((np.arange(len(vy)),-p,-rs))
            model_only=ranked[:count]
            for quota in np.linspace(0,1,11):
                reserved_count=int(round(count*quota))
                forced=ranked[flag[ranked]][:reserved_count]
                remaining=ranked[~np.isin(ranked,forced)][:count-len(forced)]
                chosen=np.concatenate([forced,remaining])
                rows.append({"repeat":j//5+1,"fold":j%5+1,
                    "budget":budget,"reserve_fraction":float(quota),
                    "policy":"rule_reserved","reviewed":count,
                    "recall":float(vy[chosen].sum()/vy.sum()),
                    "precision":float(vy[chosen].mean()),
                    "labeled_positives":int(vy[chosen].sum()),
                    "flagged_reviewed":int(flag[chosen].sum()),
                    "flagged_share":float(flag[chosen].mean()),
                    "rule_positive_coverage":float(flag[chosen].sum()/flag.sum()),
                    "replacement_count":int(count-len(set(chosen).intersection(model_only))),
                    "model_base_positive":int(vy[model_only].sum()),
                    "model_base_flagged_n":int(flag[model_only].sum()),
                    "ap":float(average_precision_score(vy,p))})
            chosen=rule_ranked[:count]
            rows.append({"repeat":j//5+1,"fold":j%5+1,"budget":budget,
                "reserve_fraction":-1.0,"policy":"rule_first","reviewed":count,
                "recall":float(vy[chosen].sum()/vy.sum()),
                "precision":float(vy[chosen].mean()),
                "labeled_positives":int(vy[chosen].sum()),
                "flagged_reviewed":int(flag[chosen].sum()),
                "flagged_share":float(flag[chosen].mean()),
                "rule_positive_coverage":float(flag[chosen].sum()/flag.sum()),
                "replacement_count":int(count-len(set(chosen).intersection(model_only))),
                "model_base_positive":int(vy[model_only].sum()),
                "model_base_flagged_n":int(flag[model_only].sum()),
                "ap":float(average_precision_score(vy,p))})
    dst=Path(out);dst.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(dst/"decision_policy_ablation_15fold.csv",index=False)
    summ={"seed":SEED,"n_provider":len(y),
      "validation":"3 repeats x 5 provider-disjoint folds, overlapping across repeats",
      "metrics":[{"budget":float(b),"policy":policy,"reserve_fraction":float(q),
         **{k:float(group[k].mean()) for k in
         ("recall","precision","flagged_share","rule_positive_coverage","replacement_count")}}
         for (b,policy,q),group in frame.groupby(("budget","policy","reserve_fraction"))]}
    (dst/"decision_policy_ablation_summary.json").write_text(json.dumps(summ,indent=2))
    print(json.dumps(summ,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",default="results")
    a=ap.parse_args()
    run(a.data,a.out)
