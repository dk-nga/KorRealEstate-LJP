"""
케이스 특성별 오류 분석.

질문
  1) 어떤 사건 유형(카테고리·시대·길이·조문 수)에서 더 틀리는가
  2) 모델들이 같은 사건에서 함께 틀리는가 (사건 난이도) vs 제각각 틀리는가 (모델 특이성)
  3) 전 모델이 실패하는 사건의 공통 특징은 무엇인가

케이스 단위 점수 = 그 사건의 12조건 평균 정확도 (모델별), 즉 조건 노이즈를 평균해 없앤 값.
"""

import json
import os
import re
import glob
import statistics as st
from collections import defaultdict

from scipy.stats import binomtest, pearsonr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data", "legal_dataset_v2_full.jsonl")
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
SHORT = {m: m.split("-")[0][:6] for m in MODELS}

# 주 실험 12조건만 센다. 반사실 스왑 결과가 results/ 에 들어오면서 사건별 조건 수가
# 12를 넘었고, "12조건 중 7개 이상 정답 = 푼 것" 이라는 난이도 정의와 모델 간 상관이
# 조용히 오염됐다 (하드케이스 61건 -> 53건). 게다가 스왑이 실행 중이면 재실행마다
# 숫자가 달라진다. 조건 집합을 명시적으로 고정한다.
MAIN_CONDS = frozenset(
  f"{hr}__{st}__{lg}"
  for hr in ("no_reasoning", "with_reasoning")
  for st in ("no_statute", "statute_cite", "statute_full")
  for lg in ("ko", "en")
)


def load():
  cases = {r["case_id"]: r for r in
           (json.loads(l) for l in open(DATA, encoding="utf-8"))}
  # per_case[model][case_id] = [correct, ...] (12조건)
  per = defaultdict(lambda: defaultdict(list))
  for f in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
    name = os.path.basename(f)
    if name.startswith("_"):
      continue
    model, cond = name.replace(".jsonl", "").split("__", 1)
    if cond not in MAIN_CONDS:
      continue
    for line in open(f, encoding="utf-8"):
      if line.strip():
        r = json.loads(line)
        per[model][r["case_id"]].append(r["correct"])
  return cases, per


def score(per, m, cid):
  v = per[m][cid]
  return sum(v) / len(v)


def group_table(cases, per, keyfn, order=None, label="group", minn=20):
  groups = defaultdict(list)
  for cid, c in cases.items():
    groups[keyfn(c)].append(cid)
  keys = order or sorted(groups, key=lambda k: -len(groups[k]))
  print(f"{label:22s} {'n':>5s} " + " ".join(f"{SHORT[m]:>7s}" for m in MODELS) + f" {'평균':>7s}")
  for k in keys:
    ids = groups.get(k, [])
    if len(ids) < minn:
      continue
    cells = [st.mean(score(per, m, cid) for cid in ids) for m in MODELS]
    print(f"{str(k):22s} {len(ids):5d} " + " ".join(f"{c:7.3f}" for c in cells)
          + f" {st.mean(cells):7.3f}")


def main():
  cases, per = load()
  ids = sorted(cases)
  print(f"사건 {len(ids)}건 | 모델별 케이스 점수 = 12조건 평균 정확도\n")

  # ── 1. 카테고리 ─────────────────────────────────────────────
  print("=" * 96)
  print("1. 사건 카테고리별")
  print("=" * 96)
  group_table(cases, per, lambda c: c["category"], label="category")

  print("\n[카테고리별 골든 파기 비율 — 난이도가 라벨 편중 때문인지 확인]")
  by = defaultdict(list)
  for cid, c in cases.items():
    by[c["category"]].append(c["label"])
  for k, v in sorted(by.items(), key=lambda x: -len(x[1])):
    if len(v) < 20:
      continue
    print(f"  {k:20s} n={len(v):4d}  파기율={sum(1 for x in v if x=='파기')/len(v):.3f}")

  # ── 2. 시대 ────────────────────────────────────────────────
  print("\n" + "=" * 96)
  print("2. 판결 시대별")
  print("=" * 96)
  group_table(cases, per, lambda c: c["era"],
              order=["~60s", "70s", "80s", "90s", "00s", "10-15", "2016+"], label="era")

  # ── 3. 입력 길이 ────────────────────────────────────────────
  print("\n" + "=" * 96)
  print("3. 입력 길이별 (사실관계+상고이유+원심판단 합계, 4분위)")
  print("=" * 96)
  lens = {cid: len(c["Factual Background"]) + len(c["Grounds for Final Appeal"])
          + len(c["the court below"]) for cid, c in cases.items()}
  qs = st.quantiles(lens.values(), n=4)

  def lenq(c):
    L = lens[c["case_id"]]
    return ("Q1 짧음" if L <= qs[0] else "Q2" if L <= qs[1]
            else "Q3" if L <= qs[2] else "Q4 김")
  group_table(cases, per, lenq, order=["Q1 짧음", "Q2", "Q3", "Q4 김"], label="length quartile")
  print(f"  (경계: {int(qs[0])} / {int(qs[1])} / {int(qs[2])}자)")

  # ── 4. 인용 조문 수 ─────────────────────────────────────────
  print("\n" + "=" * 96)
  print("4. 인용 조문 수별")
  print("=" * 96)
  nart = {cid: len(set(re.findall(r"제[0-9]+조(?:의[0-9]+)?", c["statutes_raw"])))
          for cid, c in cases.items()}

  def artq(c):
    n = nart[c["case_id"]]
    return "1개" if n <= 1 else "2개" if n == 2 else "3-4개" if n <= 4 else "5개+"
  group_table(cases, per, artq, order=["1개", "2개", "3-4개", "5개+"], label="cited articles")

  # ── 5. 골든 라벨 ────────────────────────────────────────────
  print("\n" + "=" * 96)
  print("5. 골든 라벨별 — 파기 사건이 어려운가")
  print("=" * 96)
  group_table(cases, per, lambda c: c["label"], order=["파기", "유지"], label="golden")

  # ── 6. 모델 간 오류 상관 ───────────────────────────────────
  print("\n" + "=" * 96)
  print("6. 모델 간 케이스 점수 상관 — 같은 사건에서 함께 틀리는가")
  print("=" * 96)
  print(f"{'':16s} " + " ".join(f"{SHORT[m]:>7s}" for m in MODELS))
  for m1 in MODELS:
    row = []
    for m2 in MODELS:
      r, _ = pearsonr([score(per, m1, c) for c in ids], [score(per, m2, c) for c in ids])
      row.append(r)
    print(f"{m1:16s} " + " ".join(f"{v:7.3f}" for v in row))
  print("\n높은 상관 = 사건 자체의 난이도가 지배 / 낮은 상관 = 모델별 특이 오류")

  # ── 7. 합의 난이도 ─────────────────────────────────────────
  print("\n" + "=" * 96)
  print("7. 사건 난이도 분포 — 몇 개 모델이 그 사건을 '대체로' 맞히는가")
  print("=" * 96)
  # 모델이 그 사건을 12조건 중 과반 맞히면 '맞힘'으로 간주
  solved = {cid: sum(1 for m in MODELS if score(per, m, cid) > 0.5) for cid in ids}
  dist = defaultdict(int)
  for cid in ids:
    dist[solved[cid]] += 1
  print(f"{'맞힌 모델 수':12s} {'사건 수':>7s} {'비율':>7s} {'골든파기율':>10s} {'평균길이':>9s}")
  for k in range(6):
    sel = [cid for cid in ids if solved[cid] == k]
    if not sel:
      continue
    gp = sum(1 for cid in sel if cases[cid]["label"] == "파기") / len(sel)
    ml = st.mean(lens[cid] for cid in sel)
    print(f"{k:12d} {len(sel):7d} {len(sel)/len(ids):7.3f} {gp:10.3f} {int(ml):9d}")

  print("\n[전 모델 실패 사건 (0개 모델) vs 전 모델 성공 사건 (5개 모델) 비교]")
  hard = [cid for cid in ids if solved[cid] == 0]
  easy = [cid for cid in ids if solved[cid] == 5]
  print(f"  난사건 {len(hard)}건 / 쉬운사건 {len(easy)}건")
  for nm, sel in [("난사건", hard), ("쉬운사건", easy)]:
    gp = sum(1 for cid in sel if cases[cid]["label"] == "파기") / len(sel)
    cat = defaultdict(int)
    for cid in sel:
      cat[cases[cid]["category"]] += 1
    top = sorted(cat.items(), key=lambda x: -x[1])[:3]
    print(f"  {nm}: 파기율={gp:.3f} 평균길이={int(st.mean(lens[cid] for cid in sel))} "
          f"주요카테고리={', '.join(f'{k}({v})' for k,v in top)}")
  # 난이도와 라벨의 연관성 검정
  k = sum(1 for cid in hard if cases[cid]["label"] == "파기")
  print(f"\n  난사건의 파기 편중 검정: {k}/{len(hard)} p={binomtest(k, len(hard), 0.5).pvalue:.3e}")
  k2 = sum(1 for cid in easy if cases[cid]["label"] == "파기")
  print(f"  쉬운사건의 파기 편중 검정: {k2}/{len(easy)} p={binomtest(k2, len(easy), 0.5).pvalue:.3e}")


if __name__ == "__main__":
  main()
