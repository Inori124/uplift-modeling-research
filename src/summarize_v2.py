from pathlib import Path
import argparse,json
import numpy as np
MODELS=['random_baseline','t_learner_hgb','s_learner_hgb','x_learner_hgb_crossfit']
METRICS=['auuc','qini_area','top10_policy_gain','top20_policy_gain','top30_policy_gain']
def main(pattern='results/v2_main_seed*.json',out='results/v2_summary.json'):
 fs=sorted(Path('.').glob(pattern)); ds=[json.loads(f.read_text()) for f in fs]; r={'n_runs':len(ds),'runs':[{'seed':d['model_seed'],'file':str(f)} for d,f in zip(ds,fs)],'summary':{}}
 for m in MODELS:
  r['summary'][m]={}
  for metric in METRICS:
   a=np.array([d['metrics'][m][metric] for d in ds]); r['summary'][m][metric]={'mean':float(a.mean()),'std':float(a.std(ddof=1)),'min':float(a.min()),'max':float(a.max())}
 Path(out).write_text(json.dumps(r,ensure_ascii=False,indent=2)); print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pattern',default='results/v2_main_seed*.json');p.add_argument('--out',default='results/v2_summary.json');a=p.parse_args();main(a.pattern,a.out)
