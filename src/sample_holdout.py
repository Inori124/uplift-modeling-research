"""Create a deterministic holdout disjoint from the seed-2027 1M dev sample."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
COLS=[f'f{i}' for i in range(12)]+['treatment','conversion','visit','exposure']
TOTAL=13_979_592

def main(src,out,n=300_000,seed=9090,dev_seed=2027,dev_n=1_000_000,chunksize=200_000):
    if n>=TOTAL-dev_n:raise ValueError('holdout too large')
    dev=np.random.default_rng(dev_seed).choice(TOTAL,size=dev_n,replace=False)
    excluded=np.zeros(TOTAL,dtype=bool);excluded[dev]=True
    candidates=np.flatnonzero(~excluded);rng=np.random.default_rng(seed)
    wanted=np.sort(rng.choice(candidates,size=n,replace=False)); blocks=[];start=0
    for chunk in pd.read_csv(src,usecols=COLS,chunksize=chunksize):
        end=start+len(chunk); a=np.searchsorted(wanted,start); b=np.searchsorted(wanted,end)
        if b>a:blocks.append(chunk.iloc[wanted[a:b]-start])
        start=end
    result=pd.concat(blocks,ignore_index=True)
    for c in ['treatment','conversion','visit','exposure']:result[c]=result[c].astype('int8')
    Path(out).parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False)
    print({'out':out,'rows':len(result),'treatment_rate':float(result.treatment.mean()),'conversion_rate':float(result.conversion.mean()),'dev_overlap':0})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('src');p.add_argument('--out',default='data/criteo-final-holdout-300k.csv');p.add_argument('--n',type=int,default=300000);p.add_argument('--seed',type=int,default=9090);a=p.parse_args();main(a.src,a.out,a.n,a.seed)
