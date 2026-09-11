"""汇总多个随机种子的真实数据实验结果。"""
from pathlib import Path
import argparse, json
import numpy as np

METRICS=['auuc','qini_area','uplift_at_10pct','uplift_at_20pct','uplift_at_30pct']

def main(pattern='results/real_tlearner*.json',out_path='results/real_tlearner_summary.json'):
    files=sorted(Path('.').glob(pattern))
    rows=[]
    for f in files:
        d=json.loads(f.read_text()); item={'file':str(f),'seed':d['seed']}
        item.update({m:d['tlearner'][m] for m in METRICS}); rows.append(item)
    summary={'n_runs':len(rows),'runs':rows,'summary':{}}
    for m in METRICS:
        vals=np.array([r[m] for r in rows],dtype=float)
        summary['summary'][m]={'mean':float(vals.mean()),'std_sample':float(vals.std(ddof=1)),'min':float(vals.min()),'max':float(vals.max())}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'Saved: {out}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--pattern',default='results/real_tlearner*.json'); ap.add_argument('--out',default='results/real_tlearner_summary.json'); a=ap.parse_args(); main(a.pattern,a.out)
