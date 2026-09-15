"""在同一固定切分上比较 Random、T-learner 和 S-learner。"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURES=[f"f{i}" for i in range(12)]
METRICS=['auuc','qini_area','uplift_at_10pct','uplift_at_20pct','uplift_at_30pct']

def uplift_metrics(u,t,y,p=None):
    order=np.argsort(-u); t=np.asarray(t)[order]; y=np.asarray(y)[order]
    n=len(y); p=float(t.mean()) if p is None else float(p)
    z=t*y/max(p,1e-9)-(1-t)*y/max(1-p,1e-9)
    g=np.cumsum(z)/n; q=np.arange(1,n+1)/n
    out={'auuc':float(np.trapezoid(np.r_[0,g],np.r_[0,q])), 'qini_area':float(np.trapezoid(np.r_[0,g-q*g[-1]],np.r_[0,q])),'test_treatment_rate':p}
    for k in [.1,.2,.3]: out[f'uplift_at_{int(k*100)}pct']=float(g[int(n*k)-1])
    return out

def main(path,out_path,seed=2027,test_size=.25):
    df=pd.read_csv(path,usecols=FEATURES+['treatment','conversion'])
    X=df[FEATURES].to_numpy(dtype='float32'); t=df.treatment.to_numpy(dtype='int8'); y=df.conversion.to_numpy(dtype='int8')
    idx=np.arange(len(df)); tr,te=train_test_split(idx,test_size=test_size,random_state=seed,stratify=t)
    # T-learner：处理/对照组分别训练
    t_models={}
    for label in [0,1]:
        m=make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear'))
        sub=tr[t[tr]==label]; m.fit(X[sub],y[sub]); t_models[label]=m
    u_t=t_models[1].predict_proba(X[te])[:,1]-t_models[0].predict_proba(X[te])[:,1]
    # S-learner：同一个模型，把 treatment 作为输入特征
    Xtr_s=np.c_[X[tr],t[tr]]; Xte_1=np.c_[X[te],np.ones(len(te))]; Xte_0=np.c_[X[te],np.zeros(len(te))]
    s_model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear'))
    s_model.fit(Xtr_s,y[tr]); u_s=s_model.predict_proba(Xte_1)[:,1]-s_model.predict_proba(Xte_0)[:,1]
    # X-learner：先用结果模型构造伪处理效应，再分别拟合两组效应模型
    p_train=float(t[tr].mean())
    mu0_t=t_models[0].predict_proba(X[tr])[:,1]; mu1_t=t_models[1].predict_proba(X[tr])[:,1]
    treated=tr[t[tr]==1]; control=tr[t[tr]==0]
    d1=y[treated]-mu0_t[t[tr]==1]
    d0=mu1_t[t[tr]==0]-y[control]
    tau1_model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=0.08,l2_regularization=1.0,random_state=seed)
    tau0_model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=0.08,l2_regularization=1.0,random_state=seed+1)
    tau1_model.fit(X[treated],d1); tau0_model.fit(X[control],d0)
    u_x=(1-p_train)*tau1_model.predict(X[te])+p_train*tau0_model.predict(X[te])
    rng=np.random.default_rng(seed); u_random=rng.random(len(te))
    result={'dataset':str(path),'n':len(df),'train_n':len(tr),'test_n':len(te),'feature_columns':FEATURES,'seed':seed,'treatment_rate':float(t.mean()),'conversion_rate':float(y.mean()),'treatment_rate_train':float(t[tr].mean()),'treatment_rate_test':float(t[te].mean()),'models':{'random_baseline':uplift_metrics(u_random,t[te],y[te]),'t_learner_logistic':uplift_metrics(u_t,t[te],y[te]),'s_learner_logistic':uplift_metrics(u_s,t[te],y[te]),'x_learner_hgb':uplift_metrics(u_x,t[te],y[te])}}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--out',default='results/model_comparison_seed2027.json'); ap.add_argument('--seed',type=int,default=2027); args=ap.parse_args(); main(args.path,args.out,args.seed)
