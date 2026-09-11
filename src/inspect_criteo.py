from pathlib import Path
import argparse, json
import pandas as pd

def main(path, chunksize=200_000):
    rows=0; missing={}; sums={k:0 for k in ['treatment','conversion','visit','exposure']}; uniques={k:set() for k in sums}; feature_min={}; feature_max={}
    for chunk in pd.read_csv(path,chunksize=chunksize):
        rows += len(chunk)
        for c in chunk.columns: missing[c]=missing.get(c,0)+int(chunk[c].isna().sum())
        for c in sums:
            sums[c]+=float(chunk[c].sum())
            uniques[c].update(chunk[c].dropna().unique().tolist())
        for c in [f'f{i}' for i in range(12)]:
            feature_min[c]=min(feature_min.get(c,float('inf')),float(chunk[c].min()))
            feature_max[c]=max(feature_max.get(c,float('-inf')),float(chunk[c].max()))
    result={'path':str(path),'rows':rows,'columns':[f'f{i}' for i in range(12)]+['treatment','conversion','visit','exposure'],'missing':missing,'rates':{k:sums[k]/rows for k in sums},'unique_values':{k:sorted(uniques[k]) for k in uniques},'feature_min':feature_min,'feature_max':feature_max}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    out=Path(__file__).parents[1]/'results'/'criteo_profile.json'; out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'\nSaved: {out}')
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--chunksize',type=int,default=200000); a=ap.parse_args(); main(a.path,a.chunksize)
