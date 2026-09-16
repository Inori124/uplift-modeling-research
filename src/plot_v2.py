"""Generate v2 figures from saved evaluation JSON (no model fitting)."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def load(path): return json.loads((ROOT/path).read_text())
def save(name): plt.tight_layout();plt.savefig(OUT/name,bbox_inches='tight');plt.close()
main=load('results/v2_main_seed2027.json')
fig,ax=plt.subplots(figsize=(7.3,4.2))
for key,label,color in [('random_baseline','Random draw','#999999'),('t_learner_hgb','T-learner HGB','#377eb8'),('s_learner_hgb','S-learner HGB','#e41a1c'),('x_learner_hgb_crossfit','X-learner HGB + CF','#4daf4a')]:
 c=main['curves'][key]; ax.plot(c['fraction'],c['qini'],label=label,color=color,lw=2 if key!='random_baseline' else 1.2,alpha=.9)
ax.axhline(0,color='#333',lw=.7);ax.set(xlabel='Fraction targeted',ylabel='Qini cumulative gain',title='Qini curves (v2, seed 2027 validation)');ax.legend(frameon=False);save('v2_qini_curves_seed2027.png')

fig,ax=plt.subplots(figsize=(7.2,4.1)); models=['t_learner_hgb','s_learner_hgb','x_learner_hgb_crossfit']; labels=['T-learner','S-learner','X-learner + CF']; x=np.arange(3)
means=[];errs=[]
for code in ['085','050','033','020']:
 d=load(f'results/v2_fixed_n120k_f{code}_summary.json')['summary']['x_learner_hgb_crossfit'];means.append(d['qini_area']['mean']);errs.append(d['qini_area']['std'])
ax.errorbar(['0.85','0.50','0.33','0.20'],means,yerr=errs,fmt='o-',capsize=4,color='#4daf4a',lw=2);ax.axhline(0,color='#333',lw=.7);ax.set(xlabel='Treated fraction in fixed 120k train set',ylabel='Qini area',title='X-learner robustness to treatment composition');ax.grid(axis='y',alpha=.25);save('v2_fixed_size_treatment_robustness.png')
