"""
판결예측 결과 분석: 편향 / calibration / primary_basis / 유의성 검정.

설계 노트
- 12개 조건은 모두 동일한 1147건에 대한 반복 측정이다(쌍체). 조건·모델 비교에는
  독립표본 검정이 아니라 McNemar 검정을 쓴다.
- 미복구(빈 출력) 행은 label 이 없다. 두 가지로 나눠 보고한다:
    acc        = 미복구를 오답으로 처리 (배포 관점의 실효 정확도)
    acc_parsed = 미복구를 제외 (판단 능력 자체)
"""

import json
import glob
import os
from collections import Counter, defaultdict

from scipy.stats import binomtest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
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


def load():
  """{model: {cond: [rows]}}"""
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


def mcnemar(a_correct, b_correct):
  """쌍체 이진 결과 비교. (b 우세 - a 우세) 방향과 양측 p."""
  n01 = sum(1 for x, y in zip(a_correct, b_correct) if not x and y)  # b만 맞음
  n10 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)  # a만 맞음
  if n01 + n10 == 0:
    return n01, n10, 1.0
  p = binomtest(n01, n01 + n10, 0.5).pvalue
  return n01, n10, p


def ece(rows, bins=10):
  """Expected Calibration Error — confidence 를 신뢰도로 해석했을 때의 평균 절대 오차."""
  rows = [r for r in rows if r["predicted_label"] in VALID]
  if not rows:
    return float("nan"), []
  buckets = defaultdict(list)
  for r in rows:
    c = min(max(r["confidence"], 0.0), 1.0)
    buckets[min(int(c * bins), bins - 1)].append(r)
  total = len(rows)
  e = 0.0
  table = []
  for b in sorted(buckets):
    grp = buckets[b]
    conf = sum(r["confidence"] for r in grp) / len(grp)
    acc = sum(r["correct"] for r in grp) / len(grp)
    e += len(grp) / total * abs(conf - acc)
    table.append((b / bins, (b + 1) / bins, len(grp), conf, acc))
  return e, table


def main():
  data = load()
  gold = [r["golden_label"] for r in next(iter(data[MODELS[0]].values()))]
  base = sum(1 for g in gold if g == "파기") / len(gold)
  print(f"데이터: {len(gold)}건 | 골든 파기 비율 = {base:.3f}\n")

  # ── 1. 편향 ────────────────────────────────────────────────────
  print("=" * 100)
  print("1. 편향 (12조건 통합, n=13,764)")
  print("=" * 100)
  print(f"{'model':16s} {'acc':>6s} {'acc_p':>6s} {'bal_acc':>7s} {'P(파기)':>7s} "
        f"{'R(파기)':>7s} {'R(유지)':>7s} {'미복구':>5s}  {'파기편향 p':>10s}")
  for m in MODELS:
    rows = [r for c in data[m].values() for r in c]
    n = len(rows)
    par = [r for r in rows if r["predicted_label"] in VALID]
    acc = sum(r["correct"] for r in rows) / n
    accp = sum(r["correct"] for r in par) / len(par)
    tp = sum(1 for r in par if r["golden_label"] == "파기" and r["correct"])
    tn = sum(1 for r in par if r["golden_label"] == "유지" and r["correct"])
    n_g = sum(1 for r in par if r["golden_label"] == "파기")
    n_a = len(par) - n_g
    rg, ra = tp / n_g, tn / n_a
    pgi = sum(1 for r in par if r["predicted_label"] == "파기") / len(par)
    # 예측 파기 비율이 골든 비율과 다른가
    p = binomtest(sum(1 for r in par if r["predicted_label"] == "파기"), len(par), base).pvalue
    print(f"{m:16s} {acc:6.3f} {accp:6.3f} {(rg+ra)/2:7.3f} {pgi:7.3f} "
          f"{rg:7.3f} {ra:7.3f} {n-len(par):5d}  {p:10.2e}")
  print("\nacc=미복구 오답처리 / acc_p=미복구 제외 / bal_acc=클래스별 recall 평균 / R=recall")

  # 정밀도 — 논문 Table 1 의 precision 열. recall 만으로는 "파기를 드물게 부르지만
  # 부르면 맞다" 는 gemma 의 운용 특성이 보이지 않는다.
  print(f"\n{'model':16s} {'P(파기)정밀도':>13s} {'P(유지)정밀도':>13s}")
  for m in MODELS:
    rows = [r for c in data[m] for r in data[m][c] if r["predicted_label"] in VALID]
    out = []
    for lab in ("파기", "유지"):
      pred = [r for r in rows if r["predicted_label"] == lab]
      out.append(sum(1 for r in pred if r["golden_label"] == lab) / len(pred) if pred else float("nan"))
    print(f"{m:16s} {out[0]:13.3f} {out[1]:13.3f}")


  # ── 2. Calibration ────────────────────────────────────────────
  print("\n" + "=" * 100)
  print("2. Calibration")
  print("=" * 100)
  print(f"{'model':16s} {'ECE':>6s} {'평균conf':>8s} {'acc_p':>6s} {'과신도':>7s} {'conf 고유값':>10s}")
  for m in MODELS:
    rows = [r for c in data[m].values() for r in c if r["predicted_label"] in VALID]
    e, _ = ece(rows)
    mc = sum(r["confidence"] for r in rows) / len(rows)
    ac = sum(r["correct"] for r in rows) / len(rows)
    uniq = len({round(r["confidence"], 2) for r in rows})
    print(f"{m:16s} {e:6.3f} {mc:8.3f} {ac:6.3f} {mc-ac:+7.3f} {uniq:10d}")
  print("\n과신도 = 평균 confidence − 실제 정확도 (양수면 과신)")

  print("\n[상위 2개 모델 confidence 구간별 실제 정확도]")
  for m in MODELS[:2]:
    rows = [r for c in data[m].values() for r in c if r["predicted_label"] in VALID]
    _, tbl = ece(rows)
    print(f"  {m}:")
    for lo, hi, cnt, conf, acc in tbl:
      if cnt < 30:
        continue
      print(f"    conf [{lo:.1f},{hi:.1f})  n={cnt:5d}  평균conf={conf:.3f}  실제acc={acc:.3f}  차이={conf-acc:+.3f}")

  # ── 3. primary_basis ──────────────────────────────────────────
  print("\n" + "=" * 100)
  print("3. primary_basis — 모델이 '결론에 가장 영향을 준 항목'으로 자가 보고한 것")
  print("=" * 100)

  def norm_item(s):
    s = s.strip().lower()
    for k, v in [("사실관계", "사실관계"), ("factual", "사실관계"),
                 ("원심", "원심판단"), ("court below", "원심판단"),
                 ("상고이유", "상고이유"), ("grounds", "상고이유"),
                 ("참조조문", "참조조문"), ("cited", "참조조문"), ("provision", "참조조문")]:
      if k in s:
        return v
    return "기타/미보고" if s else "미보고"

  print(f"{'model':16s} " + " ".join(f"{k:>9s}" for k in ["사실관계", "원심판단", "상고이유", "참조조문", "미보고"]))
  for m in MODELS:
    rows = [r for c in data[m].values() for r in c]
    cnt = Counter(norm_item(r.get("primary_basis_top_item", "")) for r in rows)
    tot = len(rows)
    print(f"{m:16s} " + " ".join(f"{cnt.get(k,0)/tot:9.3f}" for k in
                                 ["사실관계", "원심판단", "상고이유", "참조조문"])
          + f" {(cnt.get('미보고',0)+cnt.get('기타/미보고',0))/tot:9.3f}")

  print("\n[조문이 주어졌을 때 실제로 참조조문을 1순위 근거로 꼽는 비율]")
  print(f"{'model':16s} {'조문無':>7s} {'인용제공':>8s} {'전문제공':>8s}")
  for m in MODELS:
    out = []
    for lv in ["no_statute", "statute_cite", "statute_full"]:
      rows = [r for c, rs in data[m].items() if lv in c for r in rs]
      out.append(sum(1 for r in rows if norm_item(r.get("primary_basis_top_item", "")) == "참조조문") / len(rows))
    print(f"{m:16s} {out[0]:7.3f} {out[1]:8.3f} {out[2]:8.3f}")

  print("\n[원심판단을 1순위로 꼽은 경우 vs 아닌 경우의 정확도 — with_reasoning 조건만]")
  print(f"{'model':16s} {'원심1순위 acc':>13s} {'n':>6s} {'그외 acc':>9s} {'n':>6s} {'차이':>7s}")
  for m in MODELS:
    rows = [r for c, rs in data[m].items() if "with_reasoning" in c
            for r in rs if r["predicted_label"] in VALID]
    a = [r for r in rows if norm_item(r.get("primary_basis_top_item", "")) == "원심판단"]
    b = [r for r in rows if norm_item(r.get("primary_basis_top_item", "")) != "원심판단"]
    if not a or not b:
      continue
    aa = sum(r["correct"] for r in a) / len(a)
    bb = sum(r["correct"] for r in b) / len(b)
    print(f"{m:16s} {aa:13.3f} {len(a):6d} {bb:9.3f} {len(b):6d} {aa-bb:+7.3f}")

  # ── 4. 유의성 검정 ────────────────────────────────────────────
  print("\n" + "=" * 100)
  print("4. 유의성 검정 (쌍체 McNemar, 동일 1147건 반복측정)")
  print("=" * 100)

  print("\n[4-1] 우연(0.5) 대비 — 각 모델 12조건 통합, 이항검정")
  for m in MODELS:
    rows = [r for c in data[m].values() for r in c if r["predicted_label"] in VALID]
    k = sum(r["correct"] for r in rows)
    p = binomtest(k, len(rows), 0.5, alternative="greater").pvalue
    print(f"  {m:16s} acc_p={k/len(rows):.3f}  p={p:.3e}  {'유의' if p<0.001 else ''}")

  print("\n[4-2] 원심판단 효과 — 조문·언어 짝을 고정한 6쌍의 McNemar")
  for m in MODELS:
    tot01 = tot10 = 0
    sig = 0
    for st in ["no_statute", "statute_cite", "statute_full"]:
      for lg in ["ko", "en"]:
        a = data[m][f"no_reasoning__{st}__{lg}"]
        b = data[m][f"with_reasoning__{st}__{lg}"]
        n01, n10, p = mcnemar([r["correct"] for r in a], [r["correct"] for r in b])
        tot01 += n01
        tot10 += n10
        sig += 1 if p < 0.05 else 0
      # per-pair 출력은 생략, 합산으로 보고
    _, _, p = mcnemar([0] * tot10 + [1] * tot01, [1] * tot10 + [0] * tot01)  # 합산 근사
    p = binomtest(tot01, tot01 + tot10, 0.5).pvalue
    print(f"  {m:16s} 원심有만 정답 {tot01:4d} / 원심無만 정답 {tot10:4d}  p={p:.3e}  "
          f"(6쌍 중 {sig}개 개별 유의)")

  print("\n[4-3] 조문 효과 — 원심·언어 고정, 조문無 대비")
  for m in MODELS:
    for lv in ["statute_cite", "statute_full"]:
      t01 = t10 = 0
      for hr in ["no_reasoning", "with_reasoning"]:
        for lg in ["ko", "en"]:
          a = data[m][f"{hr}__no_statute__{lg}"]
          b = data[m][f"{hr}__{lv}__{lg}"]
          n01, n10, _ = mcnemar([r["correct"] for r in a], [r["correct"] for r in b])
          t01 += n01
          t10 += n10
      p = binomtest(t01, t01 + t10, 0.5).pvalue
      mark = "유의" if p < 0.05 else "n.s."
      print(f"  {m:16s} {lv:13s} 조문有만 정답 {t01:4d} / 조문無만 정답 {t10:4d}  p={p:.3f}  {mark}")

  print("\n[4-4] 언어 효과 — 원심·조문 고정, ko 대비 en")
  for m in MODELS:
    t01 = t10 = 0
    for hr in ["no_reasoning", "with_reasoning"]:
      for st in ["no_statute", "statute_cite", "statute_full"]:
        a = data[m][f"{hr}__{st}__ko"]
        b = data[m][f"{hr}__{st}__en"]
        n01, n10, _ = mcnemar([r["correct"] for r in a], [r["correct"] for r in b])
        t01 += n01
        t10 += n10
    p = binomtest(t01, t01 + t10, 0.5).pvalue
    print(f"  {m:16s} en만 정답 {t01:4d} / ko만 정답 {t10:4d}  p={p:.3e}  "
          f"{'유의' if p<0.05 else 'n.s.'}")

  print("\n[4-5] 모델 간 비교 — 12조건 전체를 쌍체로 (인접 순위끼리)")
  for i in range(len(MODELS) - 1):
    m1, m2 = MODELS[i], MODELS[i + 1]
    a, b = [], []
    for c in sorted(data[m1]):
      a += [r["correct"] for r in data[m1][c]]
      b += [r["correct"] for r in data[m2][c]]
    n01, n10, p = mcnemar(a, b)
    print(f"  {m1:16s} vs {m2:16s}  {m1}만 정답 {n10:5d} / {m2}만 정답 {n01:5d}  "
          f"p={p:.3e}  {'유의' if p<0.05 else 'n.s.'}")


if __name__ == "__main__":
  main()
