"""在固定 Criteo 开发集上训练 T-learner，并与随机排序比较。"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES=[f"f{i}" for i in range(12)]

def uplift_metrics(u,t,y,p=None):
    order=np.argsort(-u); t=np.asarray(t)[order]; y=np.asarray(y)[order]
    n=len(y); p=float(t.mean()) if p is None else float(p)
    z=t*y/max(p,1e-9)-(1-t)*y/max(1-p,1e-9)
    g=np.cumsum(z)/n; q=np.arange(1,n+1)/n
    auuc=float(np.trapezoid(np.r_[0,g],np.r_[0,q]))
    qini=float(np.trapezoid(np.r_[0,g-q*g[-1]],np.r_[0,q]))
    out={'auuc':auuc,'qini_area':qini,'test_treatment_rate':p}
    for k in [.1,.2,.3]: out[f'uplift_at_{int(k*100)}pct']=float(g[int(n*k)-1])
    return out

def main(path,out_path,seed=2027,test_size=.25):
    df=pd.read_csv(path,usecols=FEATURES+['treatment','conversion'])
    X=df[FEATURES].to_numpy(dtype='float32'); t=df.treatment.to_numpy(dtype='int8'); y=df.conversion.to_numpy(dtype='int8')
    idx=np.arange(len(df)); tr,te=train_test_split(idx,test_size=test_size,random_state=seed,stratify=t)
    models={}
    for label in [0,1]:
        m=LogisticRegression(max_iter=200,class_weight='balanced',solver='lbfgs')
        subset=tr[t[tr]==label]; m.fit(X[subset],y[subset]); models[label]=m
    u=models[1].predict_proba(X[te])[:,1]-models[0].predict_proba(X[te])[:,1]
    rng=np.random.default_rng(seed); random_u=rng.random(len(te))
    result={'dataset':str(path),'n':len(df),'train_n':len(tr),'test_n':len(te),'feature_columns':FEATURES,'model':'T-learner logistic regression (class_weight=balanced)','seed':seed,'treatment_rate':float(t.mean()),'conversion_rate':float(y.mean()),'treatment_rate_train':float(t[tr].mean()),'treatment_rate_test':float(t[te].mean()),'tlearner':uplift_metrics(u,t[te],y[te]),'random_baseline':uplift_metrics(random_u,t[te],y[te])}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--out',default='results/real_tlearner.json'); ap.add_argument('--seed',type=int,default=2027); args=ap.parse_args(); main(args.path,args.out,args.seed)
