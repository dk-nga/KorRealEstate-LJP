"""Camera-ready copy of make_figures.py (commit 7da7d08): only model names and %p unit labels changed."""
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
SHORT = {"gemma-4-31b-it": "Gemma-4", "qwen3.6-27b": "Qwen3.6", "exaone-3.5-32b": "EXAONE-3.5",
         "midm-2.0-base": "Midm-2.0", "gpt-oss-20b": "gpt-oss"}
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


def swap_drift(data, model, pairs):
  """스왑 쌍에서 (순 drift pp, 유효 쌍 수). 양수 = 파기 쪽으로 쏠림."""
  to_rev = to_aff = n = 0
  for a, b in pairs:
    A, B = data[model].get(a), data[model].get(b)
    if not A or not B:
      continue
    for i in set(A) & set(B):
      ra, rb = A[i]["predicted_label"], B[i]["predicted_label"]
      if ra in VALID and rb in VALID:
        n += 1
        if ra != rb:
          if rb == "파기":
            to_rev += 1
          else:
            to_aff += 1
  return (to_rev - to_aff) / n * 100, n


def prior(data, model):
  """주 12조건 통합 P(Rev) — 파싱된 행 기준 (Table 1 과 동일 규약)."""
  rows = [r for c in data[model] if "swap" not in c for r in data[model][c].values()
          if r["predicted_label"] in VALID]
  return sum(r["predicted_label"] == "파기" for r in rows) / len(rows)


def fig1(data):
  lang_pairs = [(f"{hr}__{st}__ko", f"{hr}__{st}__en")
                for hr in ["no_reasoning", "with_reasoning"] for st in STAT]
  swap_pairs = [(f"{hr}__statute_full__{lg}", f"{hr}__statute_swap__{lg}")
                for hr in ["no_reasoning", "with_reasoning"] for lg in ["ko", "en"]]

  # 부분 실행 중인 모델은 그리지 않는다 — 표본이 적으면 Δ가 요동쳐 그림이 거짓말을 한다
  MIN_N = 3000

  fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.12))
  for ax, (pairs, title, sub) in zip(axes[:2], [
      (lang_pairs, "(a) Translation (ko $\\rightarrow$ en)", "same case, machine-translated"),
      (swap_pairs, "(b) Statute swap", "target-case $\\rightarrow$ donor provisions")]):
    ax.axvspan(-3, 3, color="0.92", zorder=0)   # 표와 같은 points 단위
    ax.axvline(0, color="0.5", lw=0.6, zorder=1)
    for m in MODELS:
      r = flip_and_delta(data, m, pairs)
      if r is None:
        continue
      d, f, n = r
      if n < MIN_N:
        print(f"  [fig1] skip {SHORT[m]} on '{sub}': only {n} matched observations")
        continue
      ax.scatter(d * 100, f * 100, s=42, marker=MARK[m], color=COL[m], zorder=3,
                 label=SHORT[m], edgecolors="white", linewidths=0.5)
    ax.set_xlim(-7.5, 7.5)
    ax.set_xlabel("$\\Delta$ accuracy (%p)")
    ax.set_ylim(0, 40)
    ax.set_title(f"{title}\n{sub}", fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
  axes[0].set_ylabel("verdicts changed (%)")
  axes[1].tick_params(labelleft=False)

  # (c) 스왑 하 순 drift 는 모델 자신의 label prior 방향과 5/5 부호 일치
  ax = axes[2]
  xmax, ymax = 0.34, 5.4
  ax.fill_between([0, xmax], 0, ymax, color="0.93", zorder=0)
  ax.fill_between([-xmax, 0], -ymax, 0, color="0.93", zorder=0)
  ax.axhline(0, color="0.5", lw=0.6, zorder=1)
  ax.axvline(0, color="0.5", lw=0.6, zorder=1)
  for m in MODELS:
    x = prior(data, m) - 0.5
    y, n = swap_drift(data, m, swap_pairs)
    if n < MIN_N:
      continue
    # prior 의 95% CI 를 가로 오차막대로 함께 그린다. EXAONE 의 구간은 0 을 가로지르지
    # 않지만 폭이 다른 넷보다 훨씬 좁으므로, 구간을 그려야 5/5 부호 일치가 EXAONE 에서
    # 얼마나 약한 근거인지 보인다.
    #
    # 반드시 **사례수준** 으로 재야 한다. 12조건을 풀어 세면 n≈13,764 가 되어 구간이
    # 축 폭의 1% 로 줄고, EXAONE 도 0.5 에서 유의하게 떨어진 것처럼 그려진다 —
    # 이 논문이 스스로 경고하는 pooling 오류(§Discussion)를 그림이 저지르는 꼴이고,
    # 같은 그림 캡션의 "EXAONE 의 prior 는 균형과 구별되지 않는다" 와 모순된다.
    main = [c for c in data[m] if "swap" not in c]
    per_case = []
    for cid in data[m][main[0]]:
      vals = [1.0 if data[m][c][cid]["predicted_label"] == "파기" else 0.0
              for c in main
              if cid in data[m][c] and data[m][c][cid]["predicted_label"] in VALID]
      if vals:
        per_case.append(sum(vals) / len(vals))
    arr = np.array(per_case) - 0.5
    _rng = np.random.default_rng(42)          # §4 선언 규약: case bootstrap, seed 42
    _idx = _rng.integers(0, len(arr), (10000, len(arr)))
    _lo, _hi = np.percentile(arr[_idx].mean(axis=1), [2.5, 97.5])
    ax.errorbar(x, y, xerr=[[x - _lo], [_hi - x]], fmt="none", ecolor=COL[m],
                elinewidth=0.9, capsize=2, zorder=2)
    ax.scatter(x, y, s=42, marker=MARK[m], color=COL[m], zorder=3,
               edgecolors="white", linewidths=0.5)
  ax.set_xlim(-xmax, xmax)
  ax.set_ylim(-ymax, ymax)
  # 캡션이 축의 의미를 설명하므로 라벨은 한 줄로 — 두 줄이면 범례를 덮는다 (2026-08-03)
  ax.set_xlabel("$P(\\mathrm{Rev})-0.5$", labelpad=1)
  ax.set_ylabel("net drift under swap (%p)", labelpad=1)
  # 여기서 "95\%" 라고 쓰면 usetex 가 아니므로 역슬래시가 그대로 인쇄된다.
  ax.set_title("(c) Swap drift vs. reversal tendency\n(bars: case-level 95% CI)", fontsize=8)
  ax.grid(alpha=0.25, lw=0.4)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
             fontsize=7, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=1.0)
  fig.subplots_adjust(bottom=0.26, wspace=0.32)
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig1_flip_vs_delta_cr.{ext}"))
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
  im = ax.imshow(M, cmap="RdYlBu", aspect="auto", vmin=0.50, vmax=0.67)
  ax.set_xticks(range(len(conds)))
  ax.set_xticklabels([STAT_LBL[c.split("__")[1]] for c in conds], rotation=0)
  ax.set_yticks(range(len(MODELS)))
  ax.set_yticklabels([SHORT[m] for m in MODELS])
  # 셀 글씨는 인쇄 기준 7pt 이상이어야 읽힌다. 또 RdYlBu 양 끝(진한 남색/진한 빨강)에
  # 검은 글씨를 얹으면 대비가 사라지므로, 밝기에 따라 흰 글씨로 바꾼다.
  cmap = plt.get_cmap("RdYlBu")
  for i in range(len(MODELS)):
    for j in range(len(conds)):
      if not np.isnan(M[i, j]):
        rgba = cmap((M[i, j] - 0.50) / 0.16)
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        ax.text(j, i, f"{M[i, j]:.3f}"[1:], ha="center", va="center", fontsize=7.5,
                color="white" if lum < 0.55 else "black")
  for x in [2.5, 5.5, 8.5]:
    ax.axvline(x, color="k", lw=1.0)
  for x, lab in zip([1, 4, 7, 10],
                    ["no reasoning\nKorean", "no reasoning\nEnglish",
                     "+ reasoning\nKorean", "+ reasoning\nEnglish"]):
    ax.text(x, -0.85, lab, ha="center", va="center", fontsize=6.5)
  cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="accuracy")
  # 최고 셀이 0.667 이므로 상한 눈금은 반드시 0.67 이어야 잘리지 않는다
  cb.set_ticks([0.50, 0.55, 0.60, 0.65, 0.67])
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig2_condition_heatmap_cr.{ext}"))
  plt.close(fig)


def fig3(data):
  # \columnwidth(≈3.03in) 원폭으로 그린다 — 넓게 그려 축소하면 글꼴이 4pt 로 떨어진다
  # AUROC 막대 패널은 Table 6 의 AUROC 열과 완전히 중복이라 뺐다. 신뢰도 곡선만 남긴다
  # (버킷별 거동은 표가 담지 못한다). aurocs 는 표 6 수치 출력을 위해 계속 계산한다.
  fig, ax = plt.subplots(figsize=(3.05, 1.78))
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

  ax.grid(alpha=0.25, lw=0.4)
  ax.legend(frameon=False, loc="upper left")

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
  fig.tight_layout(pad=0.2)


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
    fig.savefig(os.path.join(FIGS, f"fig3_calibration_cr.{ext}"))
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
