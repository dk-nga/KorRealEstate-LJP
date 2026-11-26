"""
논문 figure 생성 — paper/figs/*.pdf

Fig.1 (signature)  집계 정확도 변화 vs 개별 판단 뒤집힘률 (언어 / 조문스왑 두 패널)
Fig.2              모델 × 12조건 정확도 히트맵 (조문 축의 평탄함)
Fig.3              신뢰도 보정 곡선 + AUROC

모든 수치는 results/*.jsonl 에서 직접 계산되며 하드코딩이 없다.
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
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
SHORT = {"gemma-4-31b-it": "Gemma-4-31B", "qwen3.6-27b": "Qwen3.6-27B",
         "exaone-3.5-32b": "EXAONE-3.5-32B", "midm-2.0-base": "Midm-2.0",
         "gpt-oss-20b": "gpt-oss-20B"}
MARK = {"gemma-4-31b-it": "o", "qwen3.6-27b": "s", "exaone-3.5-32b": "^",
        "midm-2.0-base": "D", "gpt-oss-20b": "v"}
COL = {"gemma-4-31b-it": "#1f77b4", "qwen3.6-27b": "#d62728", "exaone-3.5-32b": "#2ca02c",
       "midm-2.0-base": "#9467bd", "gpt-oss-20b": "#ff7f0e"}
STAT = ["no_statute", "statute_cite", "statute_full"]
STAT_LBL = {"no_statute": "none", "statute_cite": "cite", "statute_full": "cite+text"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42})


def load():
  data = defaultdict(dict)
  for f in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
    name = os.path.basename(f)
    if name.startswith("_"):
      continue
    model, cond = name.replace(".jsonl", "").split("__", 1)
    data[model][cond] = {r["case_id"]: r for r in
                         (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}
  return data


def flip_and_delta(data, model, pairs):
  """짝지어진 조건쌍 목록에 대해 (Δaccuracy, 판단 뒤집힘률, n) 반환."""
  na = nb = flips = n = 0
  ca = cb = 0
  for a, b in pairs:
    A, B = data[model].get(a), data[model].get(b)
    if not A or not B:
      continue
    for i in sorted(set(A) & set(B)):
      ra, rb = A[i], B[i]
      ca += ra["correct"]
      cb += rb["correct"]
      na += 1
      nb += 1
      if ra["predicted_label"] in VALID and rb["predicted_label"] in VALID:
        n += 1
        flips += ra["predicted_label"] != rb["predicted_label"]
  if not na or not n:
    return None
  return cb / nb - ca / na, flips / n, n


def fig1(data):
  lang_pairs = [(f"{hr}__{st}__ko", f"{hr}__{st}__en")
                for hr in ["no_reasoning", "with_reasoning"] for st in STAT]
  swap_pairs = [(f"{hr}__statute_full__{lg}", f"{hr}__statute_swap__{lg}")
                for hr in ["no_reasoning", "with_reasoning"] for lg in ["ko", "en"]]

  # 부분 실행 중인 모델은 그리지 않는다 — 표본이 적으면 Δ가 요동쳐 그림이 거짓말을 한다
  MIN_N = 3000

  fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
  for ax, (pairs, title, sub) in zip(axes, [
      (lang_pairs, "(a) Input language: Korean $\\rightarrow$ English",
       "semantics-preserving translation"),
      (swap_pairs, "(b) Cited provisions: genuine $\\rightarrow$ unrelated",
       "counterfactual statute swap")]):
    ax.axvspan(-0.03, 0.03, color="0.92", zorder=0)
    ax.axvline(0, color="0.5", lw=0.6, zorder=1)
    for m in MODELS:
      r = flip_and_delta(data, m, pairs)
      if r is None:
        continue
      d, f, n = r
      if n < MIN_N:
        print(f"  [fig1] skip {SHORT[m]} on '{sub}': only {n} matched observations")
        continue
      ax.scatter(d, f * 100, s=42, marker=MARK[m], color=COL[m], zorder=3,
                 label=SHORT[m], edgecolors="white", linewidths=0.5)
    ax.set_xlim(-0.075, 0.075)
    ax.set_xlabel("$\\Delta$ accuracy")
    ax.set_title(f"{title}\n{sub}", fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
  axes[0].set_ylabel("verdicts flipped (%)")
  axes[0].set_ylim(0, 40)
  axes[0].legend(loc="upper left", frameon=False, ncol=1)
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig1_flip_vs_delta.{ext}"))
  plt.close(fig)


def fig2(data):
  conds = [f"{hr}__{st}__{lg}" for hr in ["no_reasoning", "with_reasoning"]
           for lg in ["ko", "en"] for st in STAT]
  M = np.full((len(MODELS), len(conds)), np.nan)
  for i, m in enumerate(MODELS):
    for j, c in enumerate(conds):
      rows = data[m].get(c)
      if rows:
        M[i, j] = sum(r["correct"] for r in rows.values()) / len(rows)

  fig, ax = plt.subplots(figsize=(6.6, 2.3))
  im = ax.imshow(M, cmap="RdYlBu", aspect="auto", vmin=0.50, vmax=0.66)
  ax.set_xticks(range(len(conds)))
  ax.set_xticklabels([STAT_LBL[c.split("__")[1]] for c in conds], rotation=0)
  ax.set_yticks(range(len(MODELS)))
  ax.set_yticklabels([SHORT[m] for m in MODELS])
  for i in range(len(MODELS)):
    for j in range(len(conds)):
      if not np.isnan(M[i, j]):
        ax.text(j, i, f"{M[i, j]:.3f}"[1:], ha="center", va="center", fontsize=5.6)
  for x in [2.5, 5.5, 8.5]:
    ax.axvline(x, color="k", lw=1.0)
  for x, lab in zip([1, 4, 7, 10],
                    ["no reasoning\nKorean", "no reasoning\nEnglish",
                     "+ reasoning\nKorean", "+ reasoning\nEnglish"]):
    ax.text(x, -0.85, lab, ha="center", va="center", fontsize=6.5)
  fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="accuracy")
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig2_condition_heatmap.{ext}"))
  plt.close(fig)


def fig3(data):
  fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
  ax = axes[0]
  ax.plot([0.4, 1.0], [0.4, 1.0], ls="--", color="0.5", lw=0.8, label="perfect")
  edges = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001])
  for m in MODELS:
    rows = [r for c in data[m] if "swap" not in c for r in data[m][c].values()
            if r["predicted_label"] in VALID]
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
      sel = (conf >= lo) & (conf < hi)
      if sel.sum() >= 50:
        xs.append(conf[sel].mean())
        ys.append(corr[sel].mean())
    ax.plot(xs, ys, marker=MARK[m], color=COL[m], ms=3.5, lw=1.0, label=SHORT[m])
  ax.set_xlabel("self-reported confidence")
  ax.set_ylabel("empirical accuracy")
  ax.set_title("(a) Reliability", fontsize=8)
  ax.grid(alpha=0.25, lw=0.4)
  ax.legend(frameon=False, loc="upper left")

  ax = axes[1]
  aurocs = []
  for m in MODELS:
    rows = [r for c in data[m] if "swap" not in c for r in data[m][c].values()
            if r["predicted_label"] in VALID]
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=bool)
    order = np.argsort(conf)
    ranks = np.empty(len(conf), dtype=float)
    sc = conf[order]
    r_ = np.arange(1, len(conf) + 1, dtype=float)
    i = 0
    while i < len(sc):                      # 동점 순위 평균 처리
      j = i
      while j + 1 < len(sc) and sc[j + 1] == sc[i]:
        j += 1
      r_[i:j + 1] = (i + j + 2) / 2.0
      i = j + 1
    ranks[order] = r_
    npos, nneg = corr.sum(), (~corr).sum()
    aurocs.append((ranks[corr].sum() - npos * (npos + 1) / 2) / (npos * nneg))
  ax.bar([SHORT[m] for m in MODELS], aurocs, color=[COL[m] for m in MODELS], width=0.6)
  ax.axhline(0.5, color="k", ls="--", lw=0.8)
  ax.set_ylim(0.45, 0.62)
  ax.set_ylabel("AUROC of confidence")
  ax.set_title("(b) Discrimination", fontsize=8)
  ax.tick_params(axis="x", rotation=30)
  for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
  ax.grid(alpha=0.25, lw=0.4, axis="y")

  # 표 4(논문)의 수치는 이 함수에서 나온다 — figure 와 table 이 같은 필터를 쓰도록
  # 여기서 한 번에 계산해 출력한다 (파싱 실패 행 제외, 반사실 조건 제외).
  print(f"\n{'model':16s} {'ECE':>6s} {'meanconf':>9s} {'acc':>6s} {'overconf':>9s} {'AUROC':>6s}")
  for m, au in zip(MODELS, aurocs):
    rows = [r for c in data[m] if "swap" not in c for r in data[m][c].values()
            if r["predicted_label"] in VALID]
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    e = 0.0
    for lo, hi in zip(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.01, 0.1)):
      sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
      if sel.sum():
        e += sel.sum() / len(conf) * abs(conf[sel].mean() - corr[sel].mean())
    print(f"{SHORT[m]:16s} {e:6.3f} {conf.mean():9.3f} {corr.mean():6.3f} "
          f"{conf.mean()-corr.mean():+9.3f} {au:6.3f}")

  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig3_calibration.{ext}"))
  plt.close(fig)
  return dict(zip(MODELS, aurocs))


def main():
  os.makedirs(FIGS, exist_ok=True)
  data = load()
  fig1(data)
  fig2(data)
  au = fig3(data)
  print("figures written to", FIGS)
  print("AUROC:", {SHORT[k]: round(v, 3) for k, v in au.items()})


if __name__ == "__main__":
  main()
