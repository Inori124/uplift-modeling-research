"""汇总多个 seed 下 Random/T/S-learner 指标。"""
from pathlib import Path
import argparse, json
import numpy as np

METRICS=['auuc','qini_area','uplift_at_10pct','uplift_at_20pct','uplift_at_30pct']
MODELS=['random_baseline','t_learner_logistic','s_learner_logistic']

def main(pattern='results/model_comparison_seed*.json', out_path='results/model_comparison_summary.json'):
 files=sorted(Path('.').glob(pattern)); runs=[]
 for f in files:
  d=json.loads(f.read_text()); runs.append({'file':str(f),'seed':d['seed'],'models':d['models']})
 result={'n_runs':len(runs),'runs':runs,'summary':{}}
 for model in MODELS:
  result['summary'][model]={}
  for metric in METRICS:
   vals=np.array([x['models'][model][metric] for x in runs],dtype=float)
   result['summary'][model][metric]={'mean':float(vals.mean()),'std_sample':float(vals.std(ddof=1)),'min':float(vals.min()),'max':float(vals.max())}
 print(json.dumps(result,ensure_ascii=False,indent=2))
 out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'Saved: {out}')
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--pattern',default='results/model_comparison_seed*.json'); ap.add_argument('--out',default='results/model_comparison_summary.json'); a=ap.parse_args(); main(a.pattern,a.out)
