"""
실패 분석(error taxonomy) 표본 구축 — LLM judge 에 먹일 항목을 만든다.

핵심 자산: data 의 `full_text` 안에 대법원이 실제로 쓴 【이 유】가 1,147건 전부 들어 있다.
이것을 gold rationale 로 삼아 모델의 `reasoning_text` 와 대조하면, "어디서 어긋났는가"를
근거 있게 분류할 수 있다. 지금까지 어떤 분석도 이 필드를 쓰지 않았다.

분류 체계는 KLDBench system_design.md 의 5단계 실패 taxonomy 를 그대로 쓴다:
  S1 Authority Retrieval / S2 Holding Selection / S3 Temporal Applicability
  S4 Legal Application   / S5 Conclusion

표본 단위는 (사건, 모델) 이고 조건은 C11 (원심판단 有 + 조문 전문 + 한국어) 로 고정한다.
가장 많은 정보를 준 조건이라 5단계 전부가 평가 가능하고, 모델을 가장 유리한 상태에서
평가하게 되므로 실패가 정보 부족 탓이라는 반론을 미리 차단한다.

judge 실행에는 API 키가 필요하다. 이 스크립트는 키 없이 항목만 만들어 둔다.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data", "legal_dataset_v2_full.jsonl")
OUT = os.path.join(BASE, "outputs", "failure_judge_items.jsonl")

MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
MAIN_CONDS = [f"{hr}__{st}__{lg}"
              for hr in ["no_reasoning", "with_reasoning"]
              for st in ["no_statute", "statute_cite", "statute_full"]
              for lg in ["ko", "en"]]
FOCUS_COND = "with_reasoning__statute_full__ko"
REASON_PAT = re.compile(r"【\s*이\s*유\s*】")
MAX_REASON_CHARS = 8000   # 대법원 이유 최대 36k — judge 예산을 위해 절단, 절단 여부는 기록한다

JUDGE_SYSTEM = """You are a Korean legal expert auditing why an AI model reached the wrong \
conclusion about a Supreme Court final appeal.

You are given: (1) the case as the model saw it, (2) the model's own stated reasoning and its \
wrong prediction, and (3) the Supreme Court's actual stated reasons. Your job is to locate the \
EARLIEST stage at which the model's reasoning diverged from the Court's.

Stages (KLDBench failure taxonomy):
  S1 AUTHORITY_RETRIEVAL   The model failed to identify or invoke the legal provisions/doctrine \
that the Court treated as controlling, or invoked ones the Court did not rely on.
  S2 HOLDING_SELECTION     The model identified the right authority but attached the wrong legal \
proposition to it - it picked a different holding or issue than the one the appeal turns on.
  S3 TEMPORAL_APPLICABILITY The model applied a version of the law that does not govern the \
relevant point in time, or ignored an amendment/transitional rule the Court relied on.
  S4 LEGAL_APPLICATION     The model had the right rule and the right issue but applied the rule \
to the facts incorrectly - wrong subsumption, missed element, misweighed evidence standard.
  S5 CONCLUSION            The model's own reasoning actually supports the correct outcome, but \
it stated the opposite disposition (logical inconsistency, hedging, label confusion).

Also judge whether the model's reasoning is legally coherent on its own terms, independent of \
whether it matches the Court.

Report ONLY a single JSON object on the last line, with no text after it:
{"stage":"S1|S2|S3|S4|S5","confidence":0.00-1.00,"evidence_model":"<quote or close paraphrase \
from the model's reasoning showing the divergence>","evidence_court":"<the corresponding point \
in the Court's reasons>","coherent_but_wrong":true|false,"note":"<one sentence>"}

Before the JSON, explain your choice in 3-5 sentences. If two stages both fail, choose the \
EARLIEST one, because later failures cascade from it."""


def load_results():
  data = defaultdict(dict)
  for m in MODELS:
    for c in MAIN_CONDS:
      p = os.path.join(RESULTS, f"{m}__{c}.jsonl")
      if os.path.exists(p):
        data[m][c] = {r["case_id"]: r for r in
                      (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
  return data


def court_reason(row):
  """full_text 에서 대법원이 실제로 쓴 【이 유】 부분만 잘라낸다."""
  ft = row.get("full_text", "")
  parts = REASON_PAT.split(ft, 1)
  txt = parts[1].strip() if len(parts) > 1 else ft.strip()
  truncated = len(txt) > MAX_REASON_CHARS
  return txt[:MAX_REASON_CHARS], truncated


def build_user_prompt(case, res):
  reason, truncated = court_reason(case)
  blocks = [
    "[사건 정보 — 모델이 본 것]", "",
    "[사실관계]", case.get("Factual Background", "").strip(), "",
    "[원심의 판단]", case.get("the court below", "").strip(), "",
    "[상고이유]", case.get("Grounds for Final Appeal", "").strip(), "",
    "[참조조문]", case.get("statutes_raw", "").strip(),
    case.get("statutes_text", "").strip(), "",
    "=" * 60, "",
    f"[모델의 예측] {res['predicted_label']}  (정답: {res['golden_label']})",
    f"[모델이 밝힌 confidence] {res.get('confidence')}",
    f"[모델이 밝힌 1순위 근거] {res.get('primary_basis_top_item', '')}", "",
    "[모델의 근거 서술]", (res.get("reasoning_text") or "").strip(), "",
    "=" * 60, "",
    "[대법원이 실제로 밝힌 이유]" + ("  (분량 초과로 앞부분만 제시)" if truncated else ""),
    reason,
  ]
  return "\n".join(blocks).strip(), truncated


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stratified", type=int, default=400,
                  help="전 모델 실패 사건 외에 추가로 뽑을 층화 오답 표본 수")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out", default=OUT)
  args = ap.parse_args()

  rng = random.Random(args.seed)
  cases = {r["case_id"]: r for r in
           (json.loads(l) for l in open(DATA, encoding="utf-8"))}
  data = load_results()

  missing = [(m, c) for m in MODELS for c in MAIN_CONDS if c not in data[m]]
  if missing:
    raise SystemExit(f"[abort] 결과 파일 누락 {len(missing)}건 — 예: {missing[:3]}")

  # ── 사건 난이도: 모델이 12조건 중 7개 이상 맞히면 '푼 것' ──────────
  solved = defaultdict(set)
  for m in MODELS:
    for cid in cases:
      n = sum(1 for c in MAIN_CONDS
              if data[m][c].get(cid, {}).get("correct"))
      if n > 6:
        solved[cid].add(m)

  all_fail = sorted(cid for cid in cases if not solved[cid])
  print(f"전 모델 실패 사건: {len(all_fail)}건 ({len(all_fail)/len(cases):.1%})")
  print("  라벨 분포:", Counter(cases[c]["label"] for c in all_fail))

  # ── 1. 전 모델 실패 사건 전수 × 5모델 ────────────────────────────
  items = []
  for cid in all_fail:
    for m in MODELS:
      r = data[m][FOCUS_COND].get(cid)
      if r and not r["correct"] and r["predicted_label"] in ("파기", "유지"):
        items.append((cid, m, "all_fail"))

  # ── 2. 나머지 사건의 오답을 (모델 × 카테고리) 로 층화 표본 ───────
  pool = defaultdict(list)
  for cid, case in cases.items():
    if cid in set(all_fail):
      continue
    for m in MODELS:
      r = data[m][FOCUS_COND].get(cid)
      if r and not r["correct"] and r["predicted_label"] in ("파기", "유지"):
        pool[(m, case["category"])].append(cid)

  strata = sorted(pool)
  per = max(1, args.stratified // len(strata)) if strata else 0
  sampled = []
  for key in strata:
    ids = sorted(pool[key])
    rng.shuffle(ids)
    sampled += [(cid, key[0], "stratified") for cid in ids[:per]]
  rng.shuffle(sampled)
  items += sampled[:args.stratified]

  # ── 3. judge 항목 작성 ───────────────────────────────────────────
  n_trunc = 0
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, "w", encoding="utf-8") as f:
    for i, (cid, m, group) in enumerate(items):
      res = data[m][FOCUS_COND][cid]
      user, truncated = build_user_prompt(cases[cid], res)
      n_trunc += truncated
      f.write(json.dumps({
        "item_id": f"{i:05d}",
        "case_id": cid,
        "model": m,
        "condition": FOCUS_COND,
        "group": group,
        "category": cases[cid]["category"],
        "era": cases[cid]["era"],
        "golden_label": res["golden_label"],
        "predicted_label": res["predicted_label"],
        "model_confidence": res.get("confidence"),
        "primary_basis_top_item": res.get("primary_basis_top_item", ""),
        "court_reason_truncated": truncated,
        "judge_system": JUDGE_SYSTEM,
        "judge_user": user,
      }, ensure_ascii=False) + "\n")

  print(f"\n항목 {len(items)}건 → {args.out}")
  print("  구성:", Counter(g for _, _, g in items))
  print("  모델별:", Counter(m for _, m, _ in items))
  print(f"  대법원 이유가 {MAX_REASON_CHARS}자를 넘어 절단된 항목: {n_trunc}")
  lens = [len(json.loads(l)["judge_user"]) for l in open(args.out, encoding="utf-8")]
  lens.sort()
  print(f"  judge 프롬프트 길이(자): 중앙값 {lens[len(lens)//2]}, "
        f"p90 {lens[9*len(lens)//10]}, 최대 {lens[-1]}")
  print(f"  대략 입력 토큰 총량 추정: {sum(lens)//2:,} (한국어 ~2자/토큰 가정)")


if __name__ == "__main__":
  main()
