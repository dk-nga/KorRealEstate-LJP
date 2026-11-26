"""Figure 2 (camera-ready): aggregate vs paired effects for all four interventions.

Row 1: (a) translation ko->en, (b) statute swap target->donor, (c) swap drift vs reversal tendency
Row 2: (d) lower-court reasoning absent->present, (e) listed provisions none->cite+text

x (a,b,d,e): case-level mean accuracy difference in %p (missing verdicts count as errors)
y (a,b,d,e): share of complete pairs whose verdict changes (%)
(c): x = P(Rev) - 0.5 over the 12 main conditions; y = net drift toward Reversed under the swap, %p
Error bars: case-bootstrap 95% intervals (10,000 resamples, seed 42).
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
FIGS = os.path.join(BASE, "paper", "figs")
VALID = ("파기", "유지")
REV = "파기"
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
SHORT = {"gemma-4-31b-it": "Gemma-4", "qwen3.6-27b": "Qwen3.6", "exaone-3.5-32b": "EXAONE-3.5",
         "midm-2.0-base": "Midm-2.0", "gpt-oss-20b": "gpt-oss"}
MARK = {"gemma-4-31b-it": "o", "qwen3.6-27b": "s", "exaone-3.5-32b": "^",
        "midm-2.0-base": "D", "gpt-oss-20b": "v"}
COL = {"gemma-4-31b-it": "#1f77b4", "qwen3.6-27b": "#d62728", "exaone-3.5-32b": "#2ca02c",
       "midm-2.0-base": "#9467bd", "gpt-oss-20b": "#ff7f0e"}
HR = ["no_reasoning", "with_reasoning"]
ST = ["no_statute", "statute_cite", "statute_full"]
LG = ["ko", "en"]
B = 10_000

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42})


def load(model):
  out = {}
  for f in glob.glob(os.path.join(RESULTS, f"{model}__*.jsonl")):
    cond = os.path.basename(f)[len(model) + 2:-6]
    out[cond] = {r["case_id"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}
  return out


def pairs_for(kind):
  if kind == "lang":
    return [(f"{h}__{s}__ko", f"{h}__{s}__en") for h in HR for s in ST]
  if kind == "swap":
    return [(f"{h}__statute_full__{l}", f"{h}__statute_swap__{l}") for h in HR for l in LG]
  if kind == "reason":
    return [(f"no_reasoning__{s}__{l}", f"with_reasoning__{s}__{l}") for s in ST for l in LG]
  if kind == "prov":
    return [(f"{h}__no_statute__{l}", f"{h}__statute_full__{l}") for h in HR for l in LG]
  raise ValueError(kind)


def per_case(data, pairs):
  """Per case: mean acc diff over pairs; flips, complete pairs, signed change toward Reversed."""
  cases = sorted(set.intersection(*[set(data[a]) & set(data[b]) for a, b in pairs]))
  d, fl, cp, net = [], [], [], []
  for c in cases:
    acc_a = np.mean([data[a][c]["correct"] for a, _ in pairs])
    acc_b = np.mean([data[b][c]["correct"] for _, b in pairs])
    f = n = s = 0
    for a, b in pairs:
      la, lb = data[a][c]["predicted_label"], data[b][c]["predicted_label"]
      if la in VALID and lb in VALID:
        n += 1
        if la != lb:
          f += 1
          s += 1 if lb == REV else -1
    d.append(acc_b - acc_a); fl.append(f); cp.append(n); net.append(s)
  return np.array(d), np.array(fl), np.array(cp), np.array(net), len(pairs)


def boot(stat, n, rng):
  idx = rng.integers(0, n, size=(B, n))
  v = stat(idx)
  return np.percentile(v, [2.5, 97.5])


def summarise(data, kind, rng):
  d, fl, cp, net, k = per_case(data, pairs_for(kind))
  n = len(d)
  dm = d.mean() * 100
  fr = fl.sum() / cp.sum() * 100
  dr = net.sum() / (n * k) * 100   # net change toward Reversed per case-pair, %p
  d_ci = boot(lambda i: d[i].mean(1) * 100, n, rng)
  f_ci = boot(lambda i: fl[i].sum(1) / cp[i].sum(1) * 100, n, rng)
  r_ci = boot(lambda i: net[i].sum(1) / (n * k) * 100, n, rng)
  return dict(d=dm, d_ci=d_ci, f=fr, f_ci=f_ci, drift=dr, drift_ci=r_ci, n_pairs=int(cp.sum()))


def p_rev(data):
  main = [f"{h}__{s}__{l}" for h in HR for s in ST for l in LG]
  labs = [r["predicted_label"] for c in main for r in data[c].values() if r["predicted_label"] in VALID]
  return np.mean([l == REV for l in labs])


def scatter_panel(ax, stats, title):
  ax.axvspan(-3, 3, color="0.92", zorder=0)
  ax.axvline(0, color="0.5", lw=0.6, zorder=1)
  for m in MODELS:
    s = stats[m]
    ax.errorbar(s["d"], s["f"], xerr=[[s["d"] - s["d_ci"][0]], [s["d_ci"][1] - s["d"]]],
                yerr=[[s["f"] - s["f_ci"][0]], [s["f_ci"][1] - s["f"]]],
                fmt=MARK[m], color=COL[m], ms=5.5, mec="white", mew=0.5, elinewidth=0.7,
                capsize=0, zorder=3, label=SHORT[m])
  ax.set_xlim(-8, 8)
  ax.set_ylim(0, 40)
  ax.set_xlabel("$\\Delta$ accuracy (%p)")
  ax.set_title(title)
  ax.grid(alpha=0.25, lw=0.4)


def main():
  rng = np.random.default_rng(42)
  data = {m: load(m) for m in MODELS}
  S = {k: {m: summarise(data[m], k, rng) for m in MODELS} for k in ["lang", "swap", "reason", "prov"]}
  PR = {m: p_rev(data[m]) for m in MODELS}

  for k in S:
    for m in MODELS:
      s = S[k][m]
      print(f"{k:6s} {SHORT[m]:10s} d={s['d']:+.1f}%p [{s['d_ci'][0]:+.1f},{s['d_ci'][1]:+.1f}] "
            f"flips={s['f']:.1f}% [{s['f_ci'][0]:.1f},{s['f_ci'][1]:.1f}] drift={s['drift']:+.1f}%p "
            f"pairs={s['n_pairs']}" + (f" P(Rev)={PR[m]:.3f}" if k == "swap" else ""))
  json.dump({k: {SHORT[m]: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in S[k][m].items()}
                 for m in MODELS} for k in S} | {"p_rev": {SHORT[m]: PR[m] for m in MODELS}},
            open(os.path.join(BASE, "results", "_fig_signature_stats.json"), "w"), indent=1)

  fig = plt.figure(figsize=(6.9, 4.9))
  gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.45)
  ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
  ax_c = fig.add_subplot(gs[0, 2])
  ax_d = fig.add_subplot(gs[1, 0], sharey=ax_a); ax_e = fig.add_subplot(gs[1, 1], sharey=ax_a)
  scatter_panel(ax_a, S["lang"], "(a) Translation\nKorean $\\rightarrow$ English")
  scatter_panel(ax_b, S["swap"], "(b) Statute swap\nown $\\rightarrow$ donor provisions")
  scatter_panel(ax_d, S["reason"], "(d) Lower-court reasoning\nabsent $\\rightarrow$ present")
  scatter_panel(ax_e, S["prov"], "(e) Listed provisions\nnone $\\rightarrow$ cite+text")
  for ax in (ax_a, ax_d):
    ax.set_ylabel("verdicts changed (%)")
  for ax in (ax_b, ax_e):
    plt.setp(ax.get_yticklabels(), visible=False)

  ax = ax_c
  ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="0.5", lw=0.6)
  ax.fill_between([0, 0.4], 0, 8, color="0.92", zorder=0)
  ax.fill_between([-0.4, 0], -8, 0, color="0.92", zorder=0)
  for m in MODELS:
    s = S["swap"][m]
    ax.errorbar(PR[m] - 0.501, s["drift"], yerr=[[s["drift"] - s["drift_ci"][0]], [s["drift_ci"][1] - s["drift"]]],
                fmt=MARK[m], color=COL[m], ms=5.5, mec="white", mew=0.5, elinewidth=0.7, zorder=3)
  ax.set_xlim(-0.35, 0.35); ax.set_ylim(-7, 7)
  ax.set_xlabel("$P(\\mathrm{Rev}) - 0.501$")
  ax.set_ylabel("net drift to Rev. (%p)", labelpad=1)
  ax.set_title("(c) Swap drift vs.\nreversal tendency")
  ax.grid(alpha=0.25, lw=0.4)

  h, l = ax_a.get_legend_handles_labels()
  fig.legend(h, l, loc="center", bbox_to_anchor=(0.82, 0.27), frameon=False, ncol=1,
             title="model", title_fontsize=8)
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig1_flip_vs_delta_cr.{ext}"))
  print("saved", os.path.join(FIGS, "fig1_flip_vs_delta_cr.pdf"))


if __name__ == "__main__":
  main()
