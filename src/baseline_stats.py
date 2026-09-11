"""分块计算 Criteo treatment/control 基线统计，不把 3GB CSV 全部读入内存。"""
from pathlib import Path
import argparse, json
import pandas as pd


def summarize(path: str, chunksize: int = 200_000):
    n = {0: 0, 1: 0}
    conversion = {0: 0, 1: 0}
    visit = {0: 0, 1: 0}
    exposure = {0: 0, 1: 0}
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for t in (0, 1):
            part = chunk[chunk["treatment"] == t]
            n[t] += len(part)
            conversion[t] += int(part["conversion"].sum())
            visit[t] += int(part["visit"].sum())
            exposure[t] += int(part["exposure"].sum())
    out = {
        "rows": sum(n.values()),
        "groups": {
            "control": {"n": n[0], "conversion_rate": conversion[0] / n[0], "visit_rate": visit[0] / n[0], "exposure_rate": exposure[0] / n[0]},
            "treatment": {"n": n[1], "conversion_rate": conversion[1] / n[1], "visit_rate": visit[1] / n[1], "exposure_rate": exposure[1] / n[1]},
        },
        "naive_conversion_difference": conversion[1] / n[1] - conversion[0] / n[0],
        "note": "这是未调整的组间差异；由于 treatment 分配概率约为 0.85，正式 uplift 评估仍需独立测试集和 IPW/Qini。",
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--out", default="results/baseline_stats.json")
    args = ap.parse_args()
    result = summarize(args.path, args.chunksize)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
