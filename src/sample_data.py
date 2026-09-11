"""从大体积 Criteo CSV 中确定性、均匀抽取开发集。"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COLS=[f"f{i}" for i in range(12)]+["treatment","conversion","visit","exposure"]

def sample_file(src, out, n=1_000_000, seed=2027, chunksize=200_000):
    # 根据已检查的全量行数抽样；行号从 0 开始，不包含表头
    total = 13_979_592
    if n >= total: raise ValueError(f"n must be smaller than total rows {total}")
    rng=np.random.default_rng(seed)
    wanted=np.sort(rng.choice(total,size=n,replace=False))
    blocks=[]; start=0; first=True; pos=0
    for chunk in pd.read_csv(src, usecols=COLS, chunksize=chunksize):
        end=start+len(chunk)
        left=np.searchsorted(wanted,start,side='left'); right=np.searchsorted(wanted,end,side='left')
        if right>left:
            part=chunk.iloc[wanted[left:right]-start]
            blocks.append(part)
        start=end
    result=pd.concat(blocks,ignore_index=True)
    # 保持小文件可读，标签列显式转成整数
    for c in ["treatment","conversion","visit","exposure"]:
        result[c]=result[c].astype("int8")
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(out,index=False)
    print(f"saved: {out}")
    print(f"rows: {len(result)}")
    print(f"treatment_rate: {result.treatment.mean():.6f}")
    print(f"conversion_rate: {result.conversion.mean():.6f}")
    print(f"control_conversion_rate: {result.loc[result.treatment==0,'conversion'].mean():.6f}")
    print(f"treatment_conversion_rate: {result.loc[result.treatment==1,'conversion'].mean():.6f}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('--out',default='data/criteo-dev-1m.csv'); ap.add_argument('--n',type=int,default=1_000_000); ap.add_argument('--seed',type=int,default=2027); args=ap.parse_args(); sample_file(args.src,args.out,args.n,args.seed)
