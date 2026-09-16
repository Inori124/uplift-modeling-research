"""Run the complete development study with bounded subprocess concurrency."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(2027, 2032)
FRACTIONS = {'085': .85, '050': .5, '033': 1/3, '020': .2}


def main(suite='all', workers=2, bootstrap=200):
    jobs=[]
    for seed in SEEDS:
        base=[sys.executable, str(ROOT/'src/run_v2_experiment.py'), 'data/criteo-dev-1m.csv', '--seed', str(seed), '--threads', '2']
        if suite in ('all','main'):
            jobs.append((f'main-{seed}',base+['--bootstrap',str(bootstrap),'--out',f'results/v2_main_seed{seed}.json']))
        if suite in ('all','fixed-size'):
            for code,fraction in FRACTIONS.items():
                jobs.append((f'fixed-{code}-{seed}',base+['--bootstrap','0','--train-size','120000',
                    '--treated-fraction',str(fraction),'--out',f'results/v2_fixed_n120k_f{code}_seed{seed}.json']))
    logdir=ROOT/'data/local_logs';logdir.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', LOKY_MAX_CPU_COUNT='2')
    def run(job):
        name,cmd=job
        p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
        (logdir/f'{name}.log').write_text(p.stdout+'\n'+p.stderr)
        if p.returncode:
            raise RuntimeError(f'{name} failed (exit {p.returncode}); see {logdir/name}.log')
        return p.stdout.strip()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run,job):job[0] for job in jobs}
        for future in as_completed(futures):
            print(f'{futures[future]}: {future.result()}',flush=True)
    print(f'Completed {len(jobs)} experiments.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['all','main','fixed-size'],default='all')
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--bootstrap',type=int,default=200)
    a=p.parse_args();main(a.suite,a.workers,a.bootstrap)
