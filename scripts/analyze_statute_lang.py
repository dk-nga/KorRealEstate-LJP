"""
조문(statute) 무효과의 원인 규명 + 조문 수준(번호 vs 번호+내용) + 언어(ko/en) 심층 분석.

배경: statute_mapping_rate 는 1147건 전부 '1' 로 분산이 0 → 층화 변수로 쓸 수 없다.
대신 조문 데이터 자체를 감사해 "왜 무효과인가"를 설명한다.

핵심 가설: 인용은 사건 당시의 구법(舊法)을 가리키는데 제공된 전문은 현행법이라,
조문 전문이 도움이 안 되거나 오히려 방해가 된다.
"""

import json
import os
import re
import glob
from collections import defaultdict

from scipy.stats import binomtest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data", "legal_dataset_v2_full.jsonl")
VALID = ("파기", "유지")
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]

# 이 스크립트는 **주 실험 12조건만** 다룬다. 반사실 스왑(X01~X04) 결과가 results/ 에
# 함께 들어오면서, 조건을 무차별 로드하던 코드가 모델별 커버리지 차이로 KeyError 를
# 냈다. 재현성이 곧 논문의 주장이므로 여기서 조건 집합을 명시적으로 고정한다.
MAIN_CONDS = frozenset(
  f"{hr}__{st}__{lg}"
  for hr in ("no_reasoning", "with_reasoning")
  for st in ("no_statute", "statute_cite", "statute_full")
  for lg in ("ko", "en")
)


def load_results():
  data = defaultdict(dict)
  for f in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
    name = os.path.basename(f)
    if name.startswith("_"):
      continue
    model, cond = name.replace(".jsonl", "").split("__", 1)
    if cond not in MAIN_CONDS:
      continue
    data[model][cond] = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
  return data


def load_cases():
  return {r["case_id"]: r for r in
          (json.loads(l) for l in open(DATA, encoding="utf-8"))}


def mcnemar(a, b):
  n01 = sum(1 for x, y in zip(a, b) if not x and y)
  n10 = sum(1 for x, y in zip(a, b) if x and not y)
  p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
  return n01, n10, p


def paired(data, m, cond_a, cond_b, keep=None):
  """조건 A→B 쌍체 비교. keep: case_id 집합으로 부분집합 제한."""
  A = [r for r in data[m][cond_a] if keep is None or r["case_id"] in keep]
  B = [r for r in data[m][cond_b] if keep is None or r["case_id"] in keep]
  return mcnemar([r["correct"] for r in A], [r["correct"] for r in B])


def agg_pairs(data, m, pairs, keep=None):
  t01 = t10 = 0
  for a, b in pairs:
    n01, n10, _ = paired(data, m, a, b, keep)
    t01 += n01
    t10 += n10
  p = binomtest(t01, t01 + t10, 0.5).pvalue if t01 + t10 else 1.0
  return t01, t10, p


def main():
  data = load_results()
  cases = load_cases()

  # ── 0. 조문 데이터 감사 ──────────────────────────────────────
  print("=" * 100)
  print("0. 조문 데이터 감사 — 제공된 조문 전문은 믿을 만한가")
  print("=" * 100)
  old_law, cur_law = set(), set()
  for cid, c in cases.items():
    raw = c["statutes_raw"]
    is_old = ("구 " in raw) or ("개정되기 전" in raw)
    (old_law if is_old else cur_law).add(cid)
  print(f"구법(개정 전 버전) 인용 사건: {len(old_law)}/{len(cases)} = {len(old_law)/len(cases):.1%}")
  marked = sum(1 for cid in old_law
               if "구 " in cases[cid]["statutes_text"] or "개정되기 전" in cases[cid]["statutes_text"])
  print(f"  그런데 제공된 전문에 구법 표시가 남은 사건: {marked} "
        f"→ {len(old_law)-marked}건은 현행법 전문이 대신 제공됨 (시점 불일치)")

  # 연도대별 구법 인용 비율
  print("\n시대별 구법 인용 비율:")
  by_era = defaultdict(lambda: [0, 0])
  for cid, c in cases.items():
    e = by_era[c["era"]]
    e[1] += 1
    if cid in old_law:
      e[0] += 1
  for era in ["~60s", "70s", "80s", "90s", "00s", "10-15", "2016+"]:
    if era in by_era:
      o, n = by_era[era]
      print(f"  {era:8s} {o:4d}/{n:4d} = {o/n:5.1%}")

  # ── 1. 조문 효과를 구법/현행법으로 층화 ──────────────────────
  print("\n" + "=" * 100)
  print("1. 조문 효과의 층화 — 시점 불일치가 원인인가")
  print("=" * 100)
  print("조문無 → 조문전문 (원심·언어 4조합 합산 McNemar)\n")
  pairs_full = [(f"{hr}__no_statute__{lg}", f"{hr}__statute_full__{lg}")
                for hr in ["no_reasoning", "with_reasoning"] for lg in ["ko", "en"]]
  print(f"{'model':16s} | {'구법 인용 사건 (n=340)':>26s} | {'현행법 인용 사건 (n=807)':>26s}")
  print(f"{'':16s} | {'전문有':>7s} {'전문無':>7s} {'p':>9s} | {'전문有':>7s} {'전문無':>7s} {'p':>9s}")
  for m in MODELS:
    o01, o10, op = agg_pairs(data, m, pairs_full, old_law)
    c01, c10, cp = agg_pairs(data, m, pairs_full, cur_law)
    print(f"{m:16s} | {o01:7d} {o10:7d} {op:9.3f} | {c01:7d} {c10:7d} {cp:9.3f}")
  print("\n'전문有'=조문 전문 줬을 때만 정답 / '전문無'=안 줬을 때만 정답")

  # ── 2. 조문 수준: 번호만(cite) vs 번호+내용(full) ───────────
  print("\n" + "=" * 100)
  print("2. 조문 번호만 vs 번호+내용")
  print("=" * 100)
  print(f"\n{'model':16s} {'조문無':>7s} {'번호만':>7s} {'번호+내용':>9s} | {'번호만-무':>9s} {'내용-번호만':>11s}")
  for m in MODELS:
    def acc(lv):
      rs = [r for c, rows in data[m].items() if lv in c for r in rows]
      return sum(r["correct"] for r in rs) / len(rs)
    a0, a1, a2 = acc("no_statute"), acc("statute_cite"), acc("statute_full")
    print(f"{m:16s} {a0:7.3f} {a1:7.3f} {a2:9.3f} | {a1-a0:+9.3f} {a2-a1:+11.3f}")

  print("\n[번호만 → 번호+내용 쌍체 검정] 내용을 추가로 주는 게 도움이 되는가")
  pairs_cf = [(f"{hr}__statute_cite__{lg}", f"{hr}__statute_full__{lg}")
              for hr in ["no_reasoning", "with_reasoning"] for lg in ["ko", "en"]]
  print(f"{'model':16s} {'내용有만정답':>11s} {'번호만정답':>10s} {'p':>9s}  {'해석':>6s}")
  for m in MODELS:
    t01, t10, p = agg_pairs(data, m, pairs_cf)
    v = "내용 도움" if (p < 0.05 and t01 > t10) else ("내용 해로움" if p < 0.05 else "차이없음")
    print(f"{m:16s} {t01:11d} {t10:10d} {p:9.3f}  {v}")

  print("\n[번호+내용의 효과를 구법/현행법으로 층화] — 번호만 대비")
  print(f"{'model':16s} {'구법: 내용有/번호만':>20s} {'p':>8s} | {'현행: 내용有/번호만':>20s} {'p':>8s}")
  for m in MODELS:
    o01, o10, op = agg_pairs(data, m, pairs_cf, old_law)
    c01, c10, cp = agg_pairs(data, m, pairs_cf, cur_law)
    print(f"{m:16s} {o01:9d} /{o10:9d} {op:8.3f} | {c01:9d} /{c10:9d} {cp:8.3f}")

  # ── 3. 언어 심층 ────────────────────────────────────────────
  print("\n" + "=" * 100)
  print("3. 언어(ko/en) 심층 — 입력은 기계번역(MT)된 영어")
  print("=" * 100)

  def sub(m, lang):
    return [r for c, rows in data[m].items() if c.endswith(f"__{lang}") for r in rows]

  print(f"\n[3-1] 정확도·편향·형식·확신")
  print(f"{'model':16s} {'acc_ko':>7s} {'acc_en':>7s} {'Δ':>7s} | {'P(파기)ko':>9s} {'P(파기)en':>9s} | "
        f"{'형식ko':>7s} {'형식en':>7s} | {'conf_ko':>8s} {'conf_en':>8s}")
  for m in MODELS:
    ko, en = sub(m, "ko"), sub(m, "en")
    kp = [r for r in ko if r["predicted_label"] in VALID]
    ep = [r for r in en if r["predicted_label"] in VALID]
    ak = sum(r["correct"] for r in ko) / len(ko)
    ae = sum(r["correct"] for r in en) / len(en)
    print(f"{m:16s} {ak:7.3f} {ae:7.3f} {ae-ak:+7.3f} | "
          f"{sum(1 for r in kp if r['predicted_label']=='파기')/len(kp):9.3f} "
          f"{sum(1 for r in ep if r['predicted_label']=='파기')/len(ep):9.3f} | "
          f"{sum(1 for r in ko if r['format_ok'])/len(ko):7.3f} "
          f"{sum(1 for r in en if r['format_ok'])/len(en):7.3f} | "
          f"{sum(r['confidence'] for r in kp)/len(kp):8.3f} "
          f"{sum(r['confidence'] for r in ep)/len(ep):8.3f}")

  # [3-1b] 개별 판단 뒤집힘 — 논문 Table 4 의 flip / shift / p 열은 여기서 나온다.
  # 정확도가 거의 안 변해도 같은 사건의 판단이 언어에 따라 뒤집히는 비율을 센다.
  print(f"\n[3-1b] 개별 판단 뒤집힘 (같은 사건·같은 조건, 언어만 다름)")
  print(f"{'model':16s} {'n':>6s} {'뒤집힘':>7s} {'뒤집힘률':>9s} | "
        f"{'파기→유지':>9s} {'유지→파기':>9s} {'영어에서 쏠림':>13s} {'p':>10s}")
  for m in MODELS:
    flips = to_aff = to_rev = n = 0
    for hr in ["no_reasoning", "with_reasoning"]:
      for st in ["no_statute", "statute_cite", "statute_full"]:
        A = {r["case_id"]: r for r in data[m][f"{hr}__{st}__ko"]}
        B = {r["case_id"]: r for r in data[m][f"{hr}__{st}__en"]}
        for cid in set(A) & set(B):
          pa, pb = A[cid]["predicted_label"], B[cid]["predicted_label"]
          if pa not in VALID or pb not in VALID:
            continue
          n += 1
          if pa == pb:
            continue
          flips += 1
          if pa == "파기":
            to_aff += 1
          else:
            to_rev += 1
    p = binomtest(to_aff, to_aff + to_rev, 0.5).pvalue if flips else 1.0
    shift = "유지" if to_aff > to_rev else "파기"
    print(f"{m:16s} {n:6d} {flips:7d} {flips/n:9.1%} | {to_aff:9d} {to_rev:9d} "
          f"{shift:>13s} {p:10.2e}")
  print("p = 뒤집힘 방향이 대칭인가에 대한 이항검정")

  print(f"\n[3-2] 언어별 조문·원심 효과 — 효과가 언어에 의존하는가")
  print(f"{'model':16s} {'원심효과 ko':>12s} {'원심효과 en':>12s} | {'조문효과 ko':>12s} {'조문효과 en':>12s}")
  for m in MODELS:
    row = [m]
    for lg in ["ko", "en"]:
      prs = [(f"no_reasoning__{st}__{lg}", f"with_reasoning__{st}__{lg}")
             for st in ["no_statute", "statute_cite", "statute_full"]]
      t01, t10, p = agg_pairs(data, m, prs)
      row.append(f"{t01-t10:+5d}{'*' if p<0.05 else ' '}")
    for lg in ["ko", "en"]:
      prs = [(f"{hr}__no_statute__{lg}", f"{hr}__statute_full__{lg}")
             for hr in ["no_reasoning", "with_reasoning"]]
      t01, t10, p = agg_pairs(data, m, prs)
      row.append(f"{t01-t10:+5d}{'*' if p<0.05 else ' '}")
    print(f"{row[0]:16s} {row[1]:>12s} {row[2]:>12s} | {row[3]:>12s} {row[4]:>12s}")
  print("숫자 = (해당 조건이 도운 건수 − 해친 건수), * = p<0.05")

  print(f"\n[3-3] 사건 시대별 ko/en 정확도 — 번역 난이도가 시대와 연동되는가")
  print(f"{'era':8s} {'n':>5s} " + " ".join(f"{m.split('-')[0][:6]:>13s}" for m in MODELS))
  for era in ["~60s", "70s", "80s", "90s", "00s", "10-15", "2016+"]:
    ids = {cid for cid, c in cases.items() if c["era"] == era}
    if not ids:
      continue
    cells = []
    for m in MODELS:
      ko = [r for c, rows in data[m].items() if c.endswith("__ko")
            for r in rows if r["case_id"] in ids]
      en = [r for c, rows in data[m].items() if c.endswith("__en")
            for r in rows if r["case_id"] in ids]
      ak = sum(r["correct"] for r in ko) / len(ko)
      ae = sum(r["correct"] for r in en) / len(en)
      cells.append(f"{ak:.3f}/{ae:.3f}")
    print(f"{era:8s} {len(ids):5d} " + " ".join(f"{c:>13s}" for c in cells))
  print("각 칸 = ko/en 정확도")


if __name__ == "__main__":
  main()
