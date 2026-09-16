"""统一基学习器的 v2 uplift 实验。

主比较使用未加权 HistGradientBoostingClassifier 结果模型和
HistGradientBoostingRegressor 效应模型；X-learner 的 outcome nuisance
预测使用 training-only 3-fold cross-fitting。评估集固定为同一 validation。
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from evaluation_v2 import evaluate, paired_bootstrap

FEATURES=[f'f{i}' for i in range(12)]

def outcome(seed):
 return HistGradientBoostingClassifier(max_iter=50,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=seed)
def effect(seed):
 return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=seed)

def fit_model(path, seed=2027, max_bootstrap=100, out='results/v2_main_seed2027.json'):
 df=pd.read_csv(path,usecols=FEATURES+['treatment','conversion'])
 X=df[FEATURES].to_numpy(dtype='float32'); t=df.treatment.to_numpy(dtype='int8'); y=df.conversion.to_numpy(dtype='int8')
 idx=np.arange(len(df)); pool, val=train_test_split(idx,test_size=.25,random_state=2027,stratify=t)
 Xp,tp,yp=X[pool],t[pool],y[pool]; Xv,tv,yv=X[val],t[val],y[val]; p=float(tp.mean())
 tmodels={}
 for arm in (0,1):
  m=outcome(seed+arm); mask=tp==arm; m.fit(Xp[mask],yp[mask]); tmodels[arm]=m
 u_t=tmodels[1].predict_proba(Xv)[:,1]-tmodels[0].predict_proba(Xv)[:,1]
 sm=outcome(seed+10); sm.fit(np.c_[Xp,tp],yp); u_s=sm.predict_proba(np.c_[Xv,np.ones(len(val))])[:,1]-sm.predict_proba(np.c_[Xv,np.zeros(len(val))])[:,1]
 mu0=np.empty(len(pool)); mu1=np.empty(len(pool)); skf=StratifiedKFold(3,shuffle=True,random_state=seed)
 for fitpos,valpos in skf.split(Xp,tp):
  fold={}
  for arm in (0,1):
   m=outcome(seed+20+arm); mask=tp[fitpos]==arm; m.fit(Xp[fitpos][mask],yp[fitpos][mask]); fold[arm]=m
  mu0[valpos]=fold[0].predict_proba(Xp[valpos])[:,1]; mu1[valpos]=fold[1].predict_proba(Xp[valpos])[:,1]
 d1=yp[tp==1]-mu0[tp==1]; d0=mu1[tp==0]-yp[tp==0]
 m1=effect(seed+30); m0=effect(seed+31); m1.fit(Xp[tp==1],d1); m0.fit(Xp[tp==0],d0)
 u_x=(1-p)*m1.predict(Xv)+p*m0.predict(Xv)
 scores={'random_baseline':np.random.default_rng(seed).random(len(val)),'t_learner_hgb':u_t,'s_learner_hgb':u_s,'x_learner_hgb_crossfit':u_x}
 evaluations={name:evaluate(score,tv,yv,p=float(tv.mean())) for name,score in scores.items()}
 boot=paired_bootstrap(scores,tv,yv,n_bootstrap=max_bootstrap,seed=seed,p=float(tv.mean()))
 result={'version':'v2','dataset':str(path),'split_seed':2027,'model_seed':seed,'n':len(df),'train_pool_n':len(pool),'validation_n':len(val),'features':FEATURES,'train_propensity':p,'validation_propensity':float(tv.mean()),'outcome_model':'unweighted HistGradientBoostingClassifier','effect_model':'HistGradientBoostingRegressor','cross_fitting_folds':3,'metrics':{k:v['metrics'] for k,v in evaluations.items()},'random_expectation':evaluations['random_baseline']['random_expectation'],'bootstrap':boot}
 out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); return result
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--seed',type=int,default=2027); ap.add_argument('--out',default='results/v2_main_seed2027.json'); ap.add_argument('--bootstrap',type=int,default=100); a=ap.parse_args(); st=time.time(); r=fit_model(a.path,a.seed,a.bootstrap,a.out); print(json.dumps({'seed':a.seed,'qini':{k:v['qini_area'] for k,v in r['metrics'].items()},'elapsed_sec':round(time.time()-st,1)},ensure_ascii=False))
