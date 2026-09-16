"""Train frozen v2.1 configuration on dev data and evaluate untouched holdout once."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluation_v2 import evaluate,paired_bootstrap
from run_v2_experiment import FEATURES,fit_scores

def main(dev,holdout,out,seed=2027,bootstrap=200,threads=2):
 d=pd.read_csv(dev,usecols=FEATURES+['treatment','conversion']);h=pd.read_csv(holdout,usecols=FEATURES+['treatment','conversion'])
 X=d[FEATURES].to_numpy('float32');t=d.treatment.to_numpy('int8');y=d.conversion.to_numpy('int8');Xh=h[FEATURES].to_numpy('float32');th=h.treatment.to_numpy('int8');yh=h.conversion.to_numpy('int8')
 with threadpool_limits(threads):scores,_,folds=fit_scores(X,t,y,Xh,seed)
 p=float(th.mean());ev={k:evaluate(v,th,yh,p=p) for k,v in scores.items()}; boot=paired_bootstrap(scores,th,yh,n_bootstrap=bootstrap,seed=seed,p=p)
 r={'version':'v2.1-final-holdout','dev_dataset':Path(dev).name,'holdout_dataset':Path(holdout).name,'dev_sha256':hashlib.sha256(Path(dev).read_bytes()).hexdigest(),'holdout_sha256':hashlib.sha256(Path(holdout).read_bytes()).hexdigest(),'model_seed':seed,'train_n':len(d),'holdout_n':len(h),'features':FEATURES,'outcome_model':'unweighted HistGradientBoostingClassifier','effect_model':'HistGradientBoostingRegressor','cross_fitting_folds':3,'holdout_propensity':p,'metrics':{k:v['metrics'] for k,v in ev.items()},'curves':{k:v['curve'] for k,v in ev.items()},'bootstrap':boot,'status':'single final evaluation; holdout not used for model selection'}
 Path(out).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({'model_seed':seed,'holdout_n':len(h),'qini':{k:v['qini_area'] for k,v in r['metrics'].items()}},ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('dev');ap.add_argument('holdout');ap.add_argument('--out',default='results/final_holdout.json');ap.add_argument('--seed',type=int,default=2027);ap.add_argument('--bootstrap',type=int,default=200);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();main(a.dev,a.holdout,a.out,a.seed,a.bootstrap,a.threads)
