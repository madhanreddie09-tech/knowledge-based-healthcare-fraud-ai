"""Exploratory state-grouped robustness test. NOT external independent validation."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from claims_ai import load
def run(data,out):
 root=Path(data);X,y=load(root)
 inpatient=pd.read_csv(root/'Train_Inpatientdata.csv',usecols=['Provider','BeneID'])
 outpatient=pd.read_csv(root/'Train_Outpatientdata.csv',usecols=['Provider','BeneID'])
 beneficiary=pd.read_csv(root/'Train_Beneficiarydata.csv',usecols=['BeneID','State'])
 cl=pd.concat([inpatient,outpatient],ignore_index=True).merge(beneficiary,on='BeneID',how='left',validate='many_to_one')
 counts=cl.groupby(['Provider','State'],dropna=False).size().reset_index(name='n')
 counts['State']=counts.State.fillna(-1)
 dominant=counts.sort_values(['Provider','n','State'],ascending=[True,False,True]).drop_duplicates('Provider').set_index('Provider').State
 labels=pd.read_csv(root/'Train.csv').Provider
 providers=labels[labels.isin(cl.Provider)]
 assert len(providers)==len(X)
 state=dominant.reindex(providers.to_numpy());assert state.notna().all()
 members=cl[['Provider','BeneID']].drop_duplicates()
 rows=[]
 for fold,(a,b) in enumerate(GroupKFold(5).split(X,y,groups=state),1):
  tr,te=X.iloc[a],X.iloc[b]
  yt,ye=y.iloc[a],y.iloc[b]
  pa=set(providers.iloc[a]);pb=set(providers.iloc[b])
  shared=len(set(members.loc[members.Provider.isin(pa),'BeneID']) & set(members.loc[members.Provider.isin(pb),'BeneID']))
  models={'Logistic':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=71)),
    'Random forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=160,min_samples_leaf=5,n_jobs=2,class_weight='balanced_subsample',random_state=71))}
  for name,model in models.items():
   model.fit(tr,yt);score=model.predict_proba(te)[:,1]
   for budget in (.05,.10,.20):
    n=int(np.ceil(budget*len(ye)));idx=np.argsort(-score,kind='stable')[:n]
    rows.append(dict(fold=fold,model=name,budget=budget,n_test=len(b),positives=int(ye.sum()),test_prevalence=float(ye.mean()),n_test_states=int(state.iloc[b].nunique()),shared_beneficiaries=shared,ap=float(average_precision_score(ye,score)),auc=float(roc_auc_score(ye,score)),recall=float(ye.to_numpy()[idx].sum()/ye.sum()),precision=float(ye.to_numpy()[idx].mean())))
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 df=pd.DataFrame(rows);df.to_csv(out/'geographic_state_heldout_5fold.csv',index=False)
 summary={'scope':'Exploratory same-dataset state-grouped internal stress test; NOT an external test','states':int(state.nunique()),'provider_count':len(y),'fold_beneficiary_overlap':df[df.model.eq('Logistic')&df.budget.eq(.1)].shared_beneficiaries.tolist(),'mean_results':df.groupby(['model','budget'])[['recall','precision','ap','auc']].mean().reset_index().to_dict('records'),'limitations':['State reflects majority beneficiary State, not provider address','Many beneficiaries in both train and test groups','One dataset, PotentialFraud proxy labels','Developed after initial experiments']}
 (out/'geographic_stress_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='results');a=p.parse_args();run(a.data,a.out)
