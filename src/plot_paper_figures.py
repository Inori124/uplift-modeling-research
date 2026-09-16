"""Generate publication-style figures from saved, frozen JSON results."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.dpi':160,'savefig.dpi':220,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22})
COLORS={'random_baseline':'#8c8c8c','t_learner_hgb':'#377eb8','s_learner_hgb':'#e41a1c','x_learner_hgb_crossfit':'#4daf4a'}
LABELS={'random_baseline':'Random draw','t_learner_hgb':'T-learner HGB','s_learner_hgb':'S-learner HGB','x_learner_hgb_crossfit':'X-learner HGB + CF'}

def read(p): return json.loads((ROOT/p).read_text())
def save(name): plt.tight_layout();plt.savefig(OUT/name,bbox_inches='tight');plt.close()

# Dataset overview: separate scales make rare conversion visible.
profile=read('results/criteo_profile.json'); baseline=read('results/baseline_stats.json'); n0=baseline['groups']['control']['n']; n1=baseline['groups']['treatment']['n']
fig,axs=plt.subplots(1,2,figsize=(7.4,3.2))
axs[0].bar(['Control','Treatment'],[n0,n1],color=['#9e9e9e','#4778a8']);axs[0].set_title('Treatment allocation');axs[0].set_ylabel('Rows')
axs[1].bar(['Control','Treatment'],[baseline['groups']['control']['conversion_rate'],baseline['groups']['treatment']['conversion_rate']],color=['#9e9e9e','#4778a8']);axs[1].set_title('Conversion rate');axs[1].set_ylabel('Rate');axs[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
save('paper_data_overview.png')

# Main validation Qini curves from seed 2027.
main=read('results/v2_main_seed2027.json');fig,ax=plt.subplots(figsize=(7.2,4.1))
for key in LABELS:
 c=main['curves'][key];ax.plot(c['fraction'],c['qini'],label=LABELS[key],color=COLORS[key],lw=2 if key!='random_baseline' else 1.1)
ax.axhline(0,color='#333',lw=.8);ax.set(xlabel='Fraction targeted',ylabel='Cumulative Qini gain',title='Qini curves on development validation');ax.legend(frameon=False);save('paper_qini_validation.png')

# Final holdout point estimate + conditional percentile CI.
hold=read('results/final_holdout.json'); keys=['t_learner_hgb','s_learner_hgb','x_learner_hgb_crossfit'];fig,ax=plt.subplots(figsize=(7.2,4.1));x=np.arange(3);means=[hold['metrics'][k]['qini_area'] for k in keys];lo=[hold['bootstrap']['ci'][k]['qini_area'][0] for k in keys];hi=[hold['bootstrap']['ci'][k]['qini_area'][1] for k in keys];err=np.vstack([np.array(means)-lo,np.array(hi)-means]);ax.errorbar(x,means,yerr=err,fmt='o',ms=7,capsize=5,lw=1.6,color='#244b7a');ax.axhline(0,color='#333',lw=.8);ax.set_xticks(x,['T-learner','S-learner','X-learner + CF']);ax.set_ylabel('Qini area');ax.set_title('Final holdout Qini area with conditional 95% CI');save('paper_holdout_qini_ci.png')

# Final holdout curves.
fig,ax=plt.subplots(figsize=(7.2,4.1))
for key in ['random_baseline']+keys:
 c=hold['curves'][key];ax.plot(c['fraction'],c['qini'],label=LABELS.get(key,'Random draw'),color=COLORS.get(key,'#8c8c8c'),lw=2 if key!='random_baseline' else 1.1)
ax.axhline(0,color='#333',lw=.8);ax.set(xlabel='Fraction targeted',ylabel='Cumulative Qini gain',title='Qini curves on untouched final holdout');ax.legend(frameon=False);save('paper_qini_holdout.png')

# Fixed-size treatment composition.
fracs=['0.85','0.50','0.33','0.20']; vals=[];errs=[]
for code in ['085','050','033','020']:
 d=read(f'results/v2_fixed_n120k_f{code}_summary.json')['summary']['x_learner_hgb_crossfit']['qini_area'];vals.append(d['mean']);errs.append(d['std'])
fig,ax=plt.subplots(figsize=(7.2,4.0));ax.errorbar(fracs,vals,yerr=errs,fmt='o-',capsize=5,color='#4daf4a',lw=2);ax.axhline(0,color='#333',lw=.8);ax.set(xlabel='Treated fraction in fixed 120k train set',ylabel='Qini area',title='X-learner composition sensitivity');save('paper_treatment_fraction.png')
print('saved',len(list(OUT.glob('paper_*.png'))),'paper figures')
