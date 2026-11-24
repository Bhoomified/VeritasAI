"""Turn results.csv into a Markdown table + accuracy-vs-cost plot for the paper.
  python -m ml.report --results results.csv --out report/
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results.csv"); ap.add_argument("--out", default="report")
a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
df = pd.read_csv(a.results)
(out / "results.md").write_text(df.to_markdown(index=False))
num = lambda s: pd.to_numeric(s.astype(str).str.split("±").str[0], errors="coerce")
fig, ax = plt.subplots(figsize=(6, 4))
for col, mk in (("test_auc", "o"), ("cross0_auc", "s")):
    if col in df:
        ax.scatter(df["cpu_ms_video"], num(df[col]), marker=mk, label=col.replace("_auc", " AUC"))
for _, r in df.iterrows():
    ax.annotate(f"{r['approach']}:{r['backbone'][:10]}/{r['head']}", (r["cpu_ms_video"], num(pd.Series([r["test_auc"]]))[0]), fontsize=6)
ax.set_xscale("log"); ax.set_xlabel("CPU latency per video (ms, log scale)"); ax.set_ylabel("AUC"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "auc_vs_latency.png", dpi=200)
print((out / "results.md").read_text())
