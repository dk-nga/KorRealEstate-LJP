"""
논문에 실린 수치를 원자료에서 다시 계산해 대조한다.

동기: 초안을 쓰고 나서 전 수치를 재생산해 보니, 표 두 개는 어떤 스크립트도 만들지 않는
값이었고, 세 스크립트는 스왑 결과가 results/ 에 들어온 뒤로 아예 실행조차 되지 않았으며,
난이도 분석은 조건 수가 12를 넘으면서 조용히 오염돼 하드케이스가 61건에서 53건으로
바뀌어 있었다. 사람이 눈으로 대조하는 방식으로는 이런 것이 잡히지 않는다.

이 스크립트는 재실행만으로 PASS/FAIL 을 내고, 하나라도 어긋나면 비정상 종료한다.
EXPECTED 의 값은 논문 본문에 인쇄된 값이다 — 논문을 고치면 여기도 고쳐야 한다.
"""

import glob
import json
import os
import sys
from collections import defaultdict

from scipy.stats import binomtest, pearsonr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data", "legal_dataset_v2_full.jsonl")
VALID = ("파기", "유지")
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
MAIN_CONDS = [f"{hr}__{st}__{lg}"
              for hr in ("no_reasoning", "with_reasoning")
              for st in ("no_statute", "statute_cite", "statute_full")
              for lg in ("ko", "en")]
STATUTE_CONDS = [c for c in MAIN_CONDS if "no_statute" not in c]
STATUTE_TOKENS = ("참조조문", "Cited Provisions")
GROUNDS_TOKENS = ("상고이유", "Grounds for Final Appeal")
BELOW_TOKENS = ("원심의 판단", "원심판단", "Reasoning of the court below")

TOL = 0.0015   # 논문은 소수 셋째 자리까지 인쇄하므로 반올림 오차만 허용


def load():
  data = defaultdict(dict)
  for f in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
    name = os.path.basename(f)
    if name.startswith("_"):
      continue
    model, cond = name.replace(".jsonl", "").split("__", 1)
    if cond not in MAIN_CONDS:
      continue
    data[model][cond] = {r["case_id"]: r for r in
                         (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}
  return data


def cases():
  return {r["case_id"]: r for r in
          (json.loads(l) for l in open(DATA, encoding="utf-8"))}


RESULTS_LOG = []


def check(name, got, expected, tol=TOL):
  ok = expected is None or abs(got - expected) <= tol
  RESULTS_LOG.append((ok, name, got, expected))
  return ok


# ── §3 데이터 ────────────────────────────────────────────────────────
def audit_dataset(C):
  rev = sum(1 for c in C.values() if c["label"] == "파기")
  check("§3 majority-class baseline", max(rev, len(C) - rev) / len(C), 0.501)
  check("§3 gold reversal rate", rev / len(C), 0.501)


# ── §5 조문을 1순위 근거로 꼽은 부분집합의 정확도 ────────────────────
def audit_statute_attended(D):
  """'모델이 조문을 읽었다고 보고한 경우조차 나아지지 않는다' 의 근거."""
  print("\n[§5] 조문을 1순위 근거로 꼽았을 때 vs 아닐 때 (조문 제공 4조건)")
  print(f"{'model':16s} {'attended acc':>12s} {'n':>6s} {'other acc':>10s} {'Δ':>8s} {'p':>9s}")
  out = {}
  for m in MODELS:
    att, oth = [], []
    for c in STATUTE_CONDS:
      for r in D[m][c].values():
        item = str(r.get("primary_basis_top_item", ""))
        (att if any(t in item for t in STATUTE_TOKENS) else oth).append(r["correct"])
    if not att:
      continue
    a, o = sum(att) / len(att), sum(oth) / len(oth)
    p = binomtest(sum(att), len(att), o).pvalue
    out[m] = (a, len(att), o, a - o, p)
    print(f"{m:16s} {a:12.3f} {len(att):6d} {o:10.3f} {a-o:+8.3f} {p:9.3f}")
  return out


# ── §8 primary_basis 별 신뢰 구간 ────────────────────────────────────
def audit_basis_regimes(D):
  """'무엇이 결정했는지' 자기보고는 신뢰도 신호가 된다 — confidence 와 달리."""
  # 원심판단이 실제로 제공된 조건으로 한정한다. 그래야 두 자기보고 항목이 모두
  # 선택 가능한 상태에서의 비교가 되고, 선택지 유무가 만드는 착시가 섞이지 않는다.
  print("\n[§8] gemma: 1순위 근거별 정확도 (with_reasoning 6조건)")
  rows = [r for c in MAIN_CONDS if c.startswith("with_reasoning")
          for r in D["gemma-4-31b-it"][c].values()]
  buckets = defaultdict(list)
  for r in rows:
    item = str(r.get("primary_basis_top_item", ""))
    if any(t in item for t in GROUNDS_TOKENS):
      buckets["grounds"].append(r["correct"])
    elif any(t in item for t in BELOW_TOKENS):
      buckets["court_below"].append(r["correct"])
  for k, v in sorted(buckets.items()):
    print(f"  {k:12s} acc={sum(v)/len(v):.3f}  n={len(v)}")
  return {k: (sum(v) / len(v), len(v)) for k, v in buckets.items()}


# ── §7 언어 뒤집힘률이 층별로 고른가 (MT 잡음 반론 방어) ─────────────
def audit_flip_uniformity(D, C):
  print("\n[§7] 언어 뒤집힘률의 층별 분포 — MT 품질이 원인이면 한쪽에 몰려야 한다")
  lens = {cid: len(c.get("Factual Background", "")) + len(c.get("Grounds for Final Appeal", ""))
          for cid, c in C.items()}
  order = sorted(lens, key=lambda k: lens[k])
  q = {cid: min(3, i * 4 // len(order)) for i, cid in enumerate(order)}

  def flip_rate(ids):
    f = n = 0
    for m in MODELS:
      for hr in ("no_reasoning", "with_reasoning"):
        for st in ("no_statute", "statute_cite", "statute_full"):
          A, B = D[m][f"{hr}__{st}__ko"], D[m][f"{hr}__{st}__en"]
          for cid in ids:
            ra, rb = A.get(cid), B.get(cid)
            if not ra or not rb:
              continue
            if ra["predicted_label"] not in VALID or rb["predicted_label"] not in VALID:
              continue
            n += 1
            f += ra["predicted_label"] != rb["predicted_label"]
    return f / n if n else float("nan")

  spans = {}
  for axis, keyf in [("era", lambda c: c["era"]),
                     ("category", lambda c: c["category"]),
                     ("length quartile", lambda c: f"Q{q[c['case_id']]+1}")]:
    groups = defaultdict(list)
    for cid, c in C.items():
      groups[keyf(c)].append(cid)
    # n<10 인 잔여 범주(거래신고 2건, 종합부동산세 1건)는 비율이 잡음이라 범위에서 뺀다.
    # 넣으면 0.383 같은 값이 나와 "범주별로 고르다" 는 서술이 거짓이 된다.
    small = {k: v for k, v in groups.items() if len(v) < 10}
    rates = {k: flip_rate(v) for k, v in sorted(groups.items()) if len(v) >= 10}
    if small:
      print(f"    (제외: " + ", ".join(f"{k} n={len(v)}" for k, v in small.items()) + ")")
    lo, hi = min(rates.values()), max(rates.values())
    spans[axis] = (lo, hi)
    print(f"  {axis:16s} {lo:.3f} – {hi:.3f}   " +
          " ".join(f"{k}:{v:.3f}" for k, v in rates.items()))
  return spans


# ── §4.1 언어 방향의 모델별 쌍체 검정 (한국어 특화가 유리한가) ───────
def audit_language_direction(D):
  print("\n[§4.1] ko vs en 쌍체 McNemar — 한국어 특화 모델이 한국어를 선호하는가")
  print(f"{'model':16s} {'ko만정답':>9s} {'en만정답':>9s} {'p':>11s}  방향")
  out = {}
  for m in MODELS:
    n01 = n10 = 0
    for hr in ("no_reasoning", "with_reasoning"):
      for st in ("no_statute", "statute_cite", "statute_full"):
        A, B = D[m][f"{hr}__{st}__ko"], D[m][f"{hr}__{st}__en"]
        for cid in set(A) & set(B):
          a, b = A[cid]["correct"], B[cid]["correct"]
          if a and not b:
            n10 += 1
          elif b and not a:
            n01 += 1
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    out[m] = p
    print(f"{m:16s} {n10:9d} {n01:9d} {p:11.2e}  {'en' if n01 > n10 else 'ko'} 우세")
  return out


# ── §9 길이 shortcut 과 라벨-길이 교란 ───────────────────────────────
def audit_length(D, C):
  print("\n[§9] 길이 shortcut — 골든 라벨과 모델 예측의 상관 부호가 다른가")
  ids = sorted(C)
  gold = [1.0 if C[i]["label"] == "파기" else 0.0 for i in ids]
  g_len = [float(len(C[i].get("Grounds for Final Appeal", ""))) for i in ids]
  b_len = [float(len(C[i].get("the court below", ""))) for i in ids]
  rg = pearsonr(g_len, gold)[0]
  rb = pearsonr(b_len, gold)[0]
  print(f"  상고이유 길이 ~ 골든 파기 : {rg:+.3f}")
  print(f"  원심판단 길이 ~ 골든 파기 : {rb:+.3f}")
  preds = {}
  for m in MODELS:
    pv = []
    for i in ids:
      vals = [1.0 if D[m][c][i]["predicted_label"] == "파기" else 0.0
              for c in MAIN_CONDS if i in D[m][c]]
      pv.append(sum(vals) / len(vals) if vals else 0.0)
    preds[m] = (pearsonr(g_len, pv)[0], pearsonr(b_len, pv)[0])
    print(f"  {m:16s} 상고이유~예측파기 {preds[m][0]:+.3f} | 원심~예측파기 {preds[m][1]:+.3f}")

  print("\n[§9] 길이 사분위별 골든 파기율 (길이와 라벨의 교란)")
  order = sorted(ids, key=lambda i: len(C[i].get("Factual Background", "")) +
                 len(C[i].get("Grounds for Final Appeal", "")))
  qrates = []
  for k in range(4):
    chunk = order[k * len(order) // 4:(k + 1) * len(order) // 4]
    qrates.append(sum(1 for i in chunk if C[i]["label"] == "파기") / len(chunk))
    print(f"  Q{k+1}: {qrates[-1]:.3f}")
  return rg, rb, preds, qrates


def main():
  D, C = load(), cases()
  missing = [(m, c) for m in MODELS for c in MAIN_CONDS if c not in D[m]]
  if missing:
    raise SystemExit(f"[abort] 주 조건 결과 누락: {missing[:3]}")

  audit_dataset(C)
  att = audit_statute_attended(D)
  reg = audit_basis_regimes(D)
  spans = audit_flip_uniformity(D, C)
  langp = audit_language_direction(D)
  rg, rb, preds, qrates = audit_length(D, C)

  # 논문 본문에 인쇄된 값과 대조
  check("§5 qwen attended delta", att["qwen3.6-27b"][3], -0.024)
  check("§5 gemma attended delta", att["gemma-4-31b-it"][3], +0.052)
  check("§8 gemma grounds-driven acc", reg["grounds"][0], 0.835)
  check("§8 gemma grounds-driven n", float(reg["grounds"][1]), 1278.0, tol=0.5)
  check("§8 gemma court-below-driven acc", reg["court_below"][0], 0.613)
  check("§7 flip span era lo", spans["era"][0], 0.247, tol=0.002)
  check("§7 flip span era hi", spans["era"][1], 0.269, tol=0.002)
  check("§7 flip span category lo", spans["category"][0], 0.250, tol=0.002)
  check("§7 flip span category hi", spans["category"][1], 0.276, tol=0.002)
  check("§7 flip span length lo", spans["length quartile"][0], 0.246, tol=0.002)
  check("§4.1 exaone ko-vs-en p (log10)", __import__("math").log10(langp["exaone-3.5-32b"]), -5.33, tol=0.05)
  check("§4.1 midm ko-vs-en p (log10)", __import__("math").log10(langp["midm-2.0-base"]), -3.00, tol=0.05)
  check("§9 grounds-length ~ gold reversal", rg, +0.103, tol=0.002)
  check("§9 below-length ~ gold reversal", rb, -0.101, tol=0.002)
  check("§9 reversal rate Q1", qrates[0], 0.381, tol=0.002)
  check("§9 reversal rate Q4", qrates[3], 0.589, tol=0.002)

  print("\n" + "=" * 78)
  print("논문 수치 대조")
  print("=" * 78)
  bad = 0
  for ok, name, got, exp in RESULTS_LOG:
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} got={got:+.4f}  paper={exp}")
  print("=" * 78)
  if bad:
    print(f"{bad}건 불일치 — 논문 또는 스크립트 중 하나가 틀렸다.")
    sys.exit(1)
  print("전부 일치.")


if __name__ == "__main__":
  main()
