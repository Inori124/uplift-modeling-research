"""Criteo-style uplift experiment. Uses numpy/pandas only; synthetic fallback for smoke tests."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd


def make_data(n=30000, seed=7):
 r=np.random.default_rng(seed); X=r.normal(size=(n,12)); t=r.integers(0,2,n)
 tau=0.018*(1/(1+np.exp(-X[:,0])))+0.006*(X[:,1]>0)
 p=0.035+0.015/(1+np.exp(-X[:,2])); y0=r.binomial(1,p); y=np.maximum(y0,r.binomial(1,np.clip(p+tau,0,1))*t)
 return X,t,y

def fit_logistic(X,y,steps=500,lr=.15):
 X1=np.c_[np.ones(len(X)),X]; w=np.zeros(X1.shape[1]);
 for _ in range(steps):
  z=np.clip(X1@w,-30,30); q=1/(1+np.exp(-z)); w-=lr*(X1.T@(q-y)/len(y)+1e-4*np.r_[0,w[1:]])
 return w

def pred(X,w): return 1/(1+np.exp(-np.c_[np.ones(len(X)),X]@w))
def metrics(u,t,y):
 order=np.argsort(-u); t=t[order]; y=y[order]; n=len(y); p=float(t.mean()); z=t*y/max(p,1e-9)-(1-t)*y/max(1-p,1e-9); g=np.cumsum(z)/n; q=np.arange(1,n+1)/n; auuc=float(np.trapezoid(np.r_[0,g],np.r_[0,q])); qini=float(np.trapezoid(np.r_[0,g-q*g[-1]],np.r_[0,q])); out={'auuc':auuc,'qini_area':qini,'treatment_rate':p}
 for k in [.1,.2,.3]: out[f'uplift_at_{int(k*100)}pct']=float(g[int(n*k)-1])
 return out

def run(seed=7,n=30000,imbalance=None):
 X,t,y=make_data(n,seed); r=np.random.default_rng(seed+100); idx=r.permutation(n); tr,te=idx[:n*2//3],idx[n*2//3:];
 if imbalance:
  a=tr[t[tr]==1]; b=tr[t[tr]==0]; keep=r.choice(a,size=min(len(a),int(len(b)*imbalance)),replace=False); tr=np.r_[b,keep]
 Xtr,Xte=X[tr],X[te]; tt,yt=t[tr],y[tr];
 w0=fit_logistic(Xtr[tt==0],yt[tt==0]); w1=fit_logistic(Xtr[tt==1],yt[tt==1]); u=pred(Xte,w1)-pred(Xte,w0)
 return metrics(u,t[te],y[te])

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=30000); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--out',default='results/metrics.json'); a=ap.parse_args();
 rows=[]
 for label,ratio in [('balanced',None),('treatment_0.5',.5),('treatment_0.25',.25)]:
  x=run(a.seed,a.n,ratio); x['setting']=label; rows.append(x); print(label,x)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(rows,indent=2),encoding='utf8')
