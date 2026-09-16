"""根据已保存的 JSON 结果生成研究图表。"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).parents[1]
OUT=ROOT/'reports'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})

def load(p): return json.loads((ROOT/p).read_text())
def savefig(name): plt.tight_layout(); plt.savefig(OUT/name,bbox_inches='tight'); plt.close()

def bars(labels,means,stds,title,ylabel,name,color='#285f9e',zero=True):
 fig,ax=plt.subplots(figsize=(7.2,4.0)); x=np.arange(len(labels)); ax.bar(x,means,yerr=stds,capsize=4,color=color,alpha=.88); ax.set_xticks(x,labels,rotation=18,ha='right'); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.25)
 if zero: ax.axhline(0,color='#444',lw=.8)
 savefig(name)

weighted=load('results/model_comparison_crossfit_summary.json')['summary']
labels=['Random','T-learner','S-learner','X-learner\n(cross-fit)']; keys=['random_baseline','t_learner_logistic','s_learner_logistic','x_learner_hgb']
bars(labels,[weighted[k]['qini_area']['mean'] for k in keys],[weighted[k]['qini_area']['std_sample'] for k in keys],'Qini area by model (5 seeds)','Qini area','qini_model_comparison.png')

unweighted=load('results/model_comparison_unweighted_summary.json')['summary']
labels2=['T-learner','S-learner','X-learner']; keys2=['t_learner_logistic','s_learner_logistic','x_learner_hgb']; x=np.arange(len(labels2)); w=.36
fig,ax=plt.subplots(figsize=(7.2,4)); a=[unweighted[k]['qini_area']['mean'] for k in keys2]; b=[weighted[k]['qini_area']['mean'] for k in keys2]; ae=[unweighted[k]['qini_area']['std_sample'] for k in keys2]; be=[weighted[k]['qini_area']['std_sample'] for k in keys2]
ax.bar(x-w/2,a,w,yerr=ae,capsize=3,label='class_weight=None',color='#377eb8'); ax.bar(x+w/2,b,w,yerr=be,capsize=3,label='class_weight=balanced',color='#e78ac3'); ax.set_xticks(x,labels2); ax.set_ylabel('Qini area'); ax.set_title('Class-weight sensitivity (5 seeds)'); ax.axhline(0,color='#444',lw=.8); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False); savefig('class_weight_sensitivity.png')

ratios=['Original ~85%','1:1','1:2','1:4']; files=['results/model_comparison_crossfit_summary.json',None,None,None]; vals=[]; errs=[]
for r in ['1','0p5','0p25']:
 ps=list((ROOT/'results').glob(f'robustness_ratio{r}_seed*.json')); arr=np.array([load(p.relative_to(ROOT))['models']['x_learner_hgb']['qini_area'] for p in ps]); vals.append(float(arr.mean())); errs.append(float(arr.std(ddof=1)))
vals.insert(0,weighted['x_learner_hgb']['qini_area']['mean']); errs.insert(0,weighted['x_learner_hgb']['qini_area']['std_sample'])
bars(ratios,vals,errs,'X-learner treatment-ratio robustness','Qini area','treatment_ratio_robustness.png',color='#4daf4a')
print('saved figures to',OUT)
