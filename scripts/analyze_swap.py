"""
반사실(counterfactual) 조문 스왑 분석 — X01~X04 vs C05/C06/C11/C12.

설계: 프롬프트는 statute_full 조건과 바이트 단위로 동일하고, [참조조문] 블록의 내용만
같은 사건이 아닌 **다른 카테고리** 사건의 조문으로 교체된다 (seed 42, scripts/make_swap.py).

측정 세 가지:
  1. 정확도 변화 — 무관한 조문이 성능을 떨어뜨리는가 (증상 유무)
  2. 개별 판단 뒤집힘률 — 성능이 그대로여도 판단이 흔들리는가 (증상 없는 불안정성)
  3. primary_basis 의 '참조조문' 지목률 — 모델이 무관함을 감지하는가

출력 표는 논문 §6 의 근거이며 이 스크립트 재실행만으로 재생산된다.
"""

import glob
import json
import os
from collections import defaultdict

from scipy.stats import binomtest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
VALID = ("파기", "유지")
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
# 스왑 조건과 1:1 로 대응되는 기준 조건 (원심 유무 × 언어)
ARMS = [(hr, lg) for hr in ["no_reasoning", "with_reasoning"] for lg in ["ko", "en"]]
STATUTE_TOKENS = ("참조조문", "Cited Provisions", "cited provision")


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


def matched(data, model, arm):
  """(기준=statute_full, 스왑) 한 쌍을, 양쪽에 모두 존재하는 case_id 로 정렬해 반환."""
  hr, lg = arm
  base = data[model].get(f"{hr}__statute_full__{lg}")
  swap = data[model].get(f"{hr}__statute_swap__{lg}")
  if not base or not swap:
    return [], []
  ids = sorted(set(base) & set(swap))
  return [base[i] for i in ids], [swap[i] for i in ids]


def mcnemar(a, b):
  n01 = sum(1 for x, y in zip(a, b) if not x and y)
  n10 = sum(1 for x, y in zip(a, b) if x and not y)
  p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
  return n01, n10, p


def statute_basis_rate(rows):
  hit = sum(1 for r in rows if any(t in str(r.get("primary_basis_top_item", ""))
                                   for t in STATUTE_TOKENS))
  return hit / len(rows) if rows else float("nan")


def main():
  data = load()

  print("=" * 104)
  print("반사실 조문 스왑 — 커버리지")
  print("=" * 104)
  cover = {}
  for m in MODELS:
    n = sum(len(matched(data, m, arm)[0]) for arm in ARMS)
    cover[m] = n
    done = sum(1 for arm in ARMS if len(matched(data, m, arm)[0]) >= 1147)
    print(f"{m:16s} 짝지어진 관측 {n:6d} / 4588   완료 arm {done}/4")
  ready = [m for m in MODELS if cover[m] > 0]
  print("\n분석 대상:", ", ".join(ready) if ready else "(없음)")

  # 데이터 파일은 라벨이 뭉쳐 있다 (첫 150건 파기 96%, 다음 150건 파기 7%; 동일 라벨
  # 최장 연속 145건). 러너는 파일 순서대로 처리하므로, 부분 실행된 arm 은 사건의
  # 무작위 표본이 아니라 **앞부분 조각**이고 라벨 구성이 전체와 크게 다르다.
  # 쌍체 비교(Δ·뒤집힘)는 같은 사건 위에서 이뤄지므로 여전히 해석 가능하지만,
  # 절대 정확도와 다른 모델과의 비교는 이 조각 위에서 신뢰할 수 없다.
  partial = [m for m in ready if cover[m] < 4588]
  if partial:
    print("\n[주의] 부분 커버리지 모델: " + ", ".join(partial))
    print("       데이터가 라벨 순으로 뭉쳐 있어 앞부분 조각은 전체를 대표하지 않는다.")
    print("       아래 절대 정확도는 모델 간 비교에 쓰지 말 것 (쌍체 Δ·뒤집힘률은 유효).")

  # ── 1. 정확도: 조문無 / 진짜조문 / 무관조문 ──────────────────────
  print("\n" + "=" * 104)
  print("1. 정확도 — 무관한 조문을 넣으면 성능이 떨어지는가")
  print("=" * 104)
  print(f"{'model':16s} {'조문無':>8s} {'진짜조문':>9s} {'무관조문':>9s} | "
        f"{'Δ(무관-진짜)':>12s} {'진짜만정답':>10s} {'무관만정답':>10s} {'p':>9s}")
  for m in ready:
    b_all, s_all, none_all = [], [], []
    for arm in ARMS:
      b, s = matched(data, m, arm)
      if not b:
        continue
      ids = {r["case_id"] for r in b}
      hr, lg = arm
      none_rows = data[m].get(f"{hr}__no_statute__{lg}", {})
      none_all += [none_rows[i]["correct"] for i in ids if i in none_rows]
      b_all += [r["correct"] for r in b]
      s_all += [r["correct"] for r in s]
    n01, n10, p = mcnemar(b_all, s_all)   # n01: 스왑만 정답, n10: 진짜만 정답
    a_none = sum(none_all) / len(none_all) if none_all else float("nan")
    ab, asw = sum(b_all) / len(b_all), sum(s_all) / len(s_all)
    print(f"{m:16s} {a_none:8.3f} {ab:9.3f} {asw:9.3f} | {asw-ab:+12.3f} "
          f"{n10:10d} {n01:10d} {p:9.3f}")
  print("\n진짜조문 = statute_full (C05/C06/C11/C12), 무관조문 = statute_swap (X01~X04)")
  print("두 조건의 프롬프트는 [참조조문] 블록 내용을 빼면 완전히 동일하다.")

  # ── 2. 개별 판단 뒤집힘 ────────────────────────────────────────
  print("\n" + "=" * 104)
  print("2. 개별 판단 뒤집힘 — 성능이 그대로여도 판단은 흔들리는가")
  print("=" * 104)
  print(f"{'model':16s} {'n':>6s} {'뒤집힘':>7s} {'뒤집힘률':>9s} | "
        f"{'파기→유지':>9s} {'유지→파기':>9s} {'p(방향)':>9s}")
  for m in ready:
    flips = to_aff = to_rev = n = 0
    for arm in ARMS:
      b, s = matched(data, m, arm)
      for rb, rs in zip(b, s):
        pb, ps = rb["predicted_label"], rs["predicted_label"]
        if pb not in VALID or ps not in VALID:
          continue
        n += 1
        if pb == ps:
          continue
        flips += 1
        if pb == "파기":
          to_aff += 1
        else:
          to_rev += 1
    p = binomtest(to_aff, to_aff + to_rev, 0.5).pvalue if flips else 1.0
    print(f"{m:16s} {n:6d} {flips:7d} {flips/n:9.1%} | {to_aff:9d} {to_rev:9d} {p:9.3f}")
  print("\np(방향) = 뒤집힘이 한쪽으로 쏠렸는지 (이항검정, 귀무가설 = 대칭)")

  # ── 3. 조문을 근거로 지목하는 비율 ─────────────────────────────
  print("\n" + "=" * 104)
  print("3. primary_basis 가 '참조조문' 인 비율 — 모델은 무관함을 감지하는가")
  print("=" * 104)
  print(f"{'model':16s} {'진짜조문':>9s} {'무관조문':>9s} {'Δ':>9s}")
  for m in ready:
    b_all, s_all = [], []
    for arm in ARMS:
      b, s = matched(data, m, arm)
      b_all += b
      s_all += s
    rb, rs = statute_basis_rate(b_all), statute_basis_rate(s_all)
    print(f"{m:16s} {rb:9.1%} {rs:9.1%} {rs-rb:+9.1%}")
  print("\n비율이 크게 떨어지면 = 무관함을 감지함 (그런데도 정확도·판단은 보호되지 않음)")


if __name__ == "__main__":
  main()
