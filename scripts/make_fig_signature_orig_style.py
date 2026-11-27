"""Figure 2 and appendix Figure 6 in the style of the submitted Figure 2.

Same layout, titles, markers and axes as the submission; only the model names (Gemma-4, ...)
and the unit labels (points/pp -> %p) change. Numbers come from make_fig_signature.summarise.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import make_fig_signature as fs

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 8.8,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
MS = 9


def flip_panel(ax, stats, title):
  ax.axvspan(-3, 3, color="0.92", zorder=0)
  ax.axvline(0, color="0.5", lw=0.7, zorder=1)
  for m in fs.MODELS:
    s = stats[m]
    ax.plot(s["d"], s["f"], fs.MARK[m], color=fs.COL[m], ms=MS, mec="white", mew=0.6, zorder=3)
  ax.set_xlim(-7.5, 7.5); ax.set_ylim(0, 40); ax.set_yticks([0, 10, 20, 30, 40])
  ax.set_xlabel("$\\Delta$ accuracy (%p)"); ax.set_title(title)
  ax.grid(alpha=0.3, lw=0.4)


def legend(fig, y):
  h = [Line2D([], [], ls="", marker=fs.MARK[m], color=fs.COL[m], ms=8, mec="white", mew=0.5) for m in fs.MODELS]
  fig.legend(h, [fs.SHORT[m] for m in fs.MODELS], loc="lower center", bbox_to_anchor=(0.5, y),
             ncol=5, frameon=False, handletextpad=0.3, columnspacing=1.6)


def main():
  rng = np.random.default_rng(42)
  data = {m: fs.load(m) for m in fs.MODELS}
  S = {k: {m: fs.summarise(data[m], k, rng) for m in fs.MODELS} for k in ["lang", "swap", "reason", "prov"]}
  PR = {m: fs.p_rev(data[m]) for m in fs.MODELS}

  fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.8, 2.7), gridspec_kw={"wspace": 0.42})
  flip_panel(a, S["lang"], "(a) Translation (ko $\\rightarrow$ en)\nsame case, machine-translated")
  flip_panel(b, S["swap"], "(b) Statute swap\ntarget-case $\\rightarrow$ donor provisions")
  a.set_ylabel("verdicts changed (%)")
  b.set_yticklabels([])
  c.axhline(0, color="0.5", lw=0.7); c.axvline(0, color="0.5", lw=0.7)
  c.fill_between([0, 0.4], 0, 6, color="0.92", zorder=0)
  c.fill_between([-0.4, 0], -6, 0, color="0.92", zorder=0)
  for m in fs.MODELS:
    s = S["swap"][m]
    c.errorbar(PR[m] - 0.501, s["drift"], yerr=[[s["drift"] - s["drift_ci"][0]], [s["drift_ci"][1] - s["drift"]]],
               fmt=fs.MARK[m], color=fs.COL[m], ms=MS, mec="white", mew=0.6, elinewidth=0.8, capsize=0, zorder=3)
  c.set_xlim(-0.35, 0.33); c.set_ylim(-6, 6); c.set_yticks([-4, -2, 0, 2, 4])
  c.set_xlabel("$P(\\mathrm{Rev}) - 0.5$"); c.set_ylabel("net drift under swap (%p)")
  c.set_title("(c) Swap drift vs. reversal tendency\n(bars: case-level 95% CI)")
  c.grid(alpha=0.3, lw=0.4)
  legend(fig, -0.16)
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(fs.FIGS, f"fig1_flip_vs_delta_cr.{ext}"))
  plt.close(fig)

  fig, (d, e) = plt.subplots(1, 2, figsize=(5.0, 2.65), gridspec_kw={"wspace": 0.12})
  flip_panel(d, S["reason"], "(a) Lower-court reasoning\nabsent $\\rightarrow$ present")
  flip_panel(e, S["prov"], "(b) Listed provisions\nnone $\\rightarrow$ cite+text")
  d.set_ylabel("verdicts changed (%)"); e.set_yticklabels([])
  h = [Line2D([], [], ls="", marker=fs.MARK[m], color=fs.COL[m], ms=8, mec="white", mew=0.5) for m in fs.MODELS]
  fig.legend(h, [fs.SHORT[m] for m in fs.MODELS], loc="lower center", bbox_to_anchor=(0.5, -0.2),
             ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2)
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(fs.FIGS, f"figA_reason_prov_cr.{ext}"))
  plt.close(fig)
  print("saved")


if __name__ == "__main__":
  main()
