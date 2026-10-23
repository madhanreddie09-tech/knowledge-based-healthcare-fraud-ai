"""Capacity-aware review benchmark; use labels as proxy, not adjudicated fraud."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from claims_ai import load

def run(data,out):
 X,y=load(data)
 cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=71)
 rows=[]; audits=[]
 for fold,(ti,vi) in enumerate(cv.split(X,y)):
  tr,te=X.iloc[ti],X.iloc[vi]
  yt,ye=y.iloc[ti].to_numpy(),y.iloc[vi].to_numpy()
  quantiles={c:float(tr[c].quantile(.9)) for c in ['claims_per_patient','mean_reimbursement','inpatient_fraction']}
  heuristic=np.column_stack([(te[c]>q).to_numpy() for c,q in quantiles.items()]).any(axis=1)
  estimators={
   'Logistic':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=71)),
   'Random forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=160,min_samples_leaf=5,n_jobs=2,class_weight='balanced_subsample',random_state=71))}
  for name,est in estimators.items():
   est.fit(tr,yt);p=est.predict_proba(te)[:,1]
   for budget in (.05,.10,.20):
    count=max(1,int(np.ceil(len(ye)*budget)))
    selected=np.argsort(-p,kind='stable')[:count]
    precision=float(ye[selected].mean())
    rows.append(dict(repeat=fold//5+1,fold=fold%5+1,model=name,budget=budget,selected=count,positives=int(ye.sum()),captured=int(ye[selected].sum()),precision_at_budget=precision,recall_at_budget=float(ye[selected].sum()/ye.sum()),lift_over_prevalence=float(precision/ye.mean()),ap=float(average_precision_score(ye,p)),roc_auc=float(roc_auc_score(ye,p))))
   if name=='Logistic':
    high=p>=np.quantile(p,.9)
    disagreement=high!=heuristic
    audits.append(dict(repeat=fold//5+1,fold=fold%5+1,flag_rate_model=float(high.mean()),flag_rate_rules=float(heuristic.mean()),disagreement_rate=float(disagreement.mean()),disagreement_labeled_positive_rate=float(ye[disagreement].mean()) if disagreement.any() else None,agreement_labeled_positive_rate=float(ye[~disagreement].mean()) if (~disagreement).any() else None))
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 frame=pd.DataFrame(rows);audit=pd.DataFrame(audits)
 frame.to_csv(out/'policy_capacity_validation.csv',index=False)
 audit.to_csv(out/'knowledge_disagreement_audit.csv',index=False)
 summary={'dataset_providers':len(X),'positive_prevalence':float(y.mean()),'resampling':'3 repeats x 5 provider-disjoint folds (correlated)','model_budget_summary':frame.groupby(['model','budget'])[['precision_at_budget','recall_at_budget','lift_over_prevalence','ap','roc_auc']].mean().reset_index().to_dict('records'),'disagreement_summary':audit.mean(numeric_only=True).to_dict(),'limitations':'Labels are PotentialFraud proxies; no independent temporal/external tests or expert rule validation.'}
 (out/'decision_audit_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--out',default='results');a=parser.parse_args()
 run(a.data,a.out)
