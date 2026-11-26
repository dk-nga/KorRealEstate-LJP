"""
저장된 결과를 새 파서로 소급 재파싱.

배경: 초기 파서는 JSON 이 한 줄에 완결돼 있어야만 읽을 수 있었다. 모델이 닫는
중괄호를 빠뜨리거나(midm 다수) JSON 을 여러 줄에 나눠 쓰면 label 을 못 뽑아
빈 문자열로 저장 → 자동 오답 처리됐다. 파싱 실패 시 전체 출력이 reasoning_text
에 통째로 보존되므로 재실행 없이 복구 가능하다.

원본은 results/raw_v1/ 로 백업하고 results/*.jsonl 을 갱신한다.
추가 필드:
  format_ok  — 첫 파서가 그대로 읽을 수 있었는가 (모델의 형식 준수 지표)
  recovered  — 이번 재파싱으로 label 을 복구했는가
"""

import json
import os
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from output_parser import parse_prediction  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
BACKUP = os.path.join(RESULTS, "raw_v1")
VALID = ("파기", "유지")


def main():
  files = sorted(
    f for f in os.listdir(RESULTS)
    if f.endswith(".jsonl") and not f.startswith("_")
  )
  os.makedirs(BACKUP, exist_ok=True)

  agg = defaultdict(lambda: {"n": 0, "before": 0, "after": 0, "bad": 0, "rec": 0, "err": 0})

  for fname in files:
    path = os.path.join(RESULTS, fname)
    shutil.copy2(path, os.path.join(BACKUP, fname))

    with open(path, encoding="utf-8") as f:
      rows = [json.loads(l) for l in f if l.strip()]

    model = fname.split("__")[0]
    a = agg[model]

    for r in rows:
      a["n"] += 1
      was_ok = r["predicted_label"] in VALID
      r["format_ok"] = was_ok
      r["recovered"] = False
      if was_ok:
        a["before"] += 1 if r["correct"] else 0
        a["after"] += 1 if r["correct"] else 0
        continue

      a["bad"] += 1
      if r["reasoning_text"].startswith("ERROR:"):
        a["err"] += 1
        continue

      p = parse_prediction(r["reasoning_text"], r["lang"])
      if p["predicted_label"] in VALID:
        a["rec"] += 1
        r["recovered"] = True
        r["predicted_label"] = p["predicted_label"]
        r["confidence"] = p["confidence"]
        r["reasoning_text"] = p["reasoning_text"]
        r["primary_basis_json"] = p["primary_basis_json"]
        r["primary_basis_top_item"] = p["primary_basis_top_item"]
        r["primary_basis_top_reason"] = p["primary_basis_top_reason"]
        r["correct"] = r["predicted_label"] == r["golden_label"]
        a["after"] += 1 if r["correct"] else 0

    with open(path, "w", encoding="utf-8") as f:
      for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

  print(f"{'model':16s} {'n':>6s} {'unparsed':>8s} {'recovered':>9s} {'error':>5s} "
        f"{'acc_before':>10s} {'acc_after':>9s} {'delta':>6s}")
  for m, a in sorted(agg.items()):
    b, af = a["before"] / a["n"], a["after"] / a["n"]
    print(f"{m:16s} {a['n']:6d} {a['bad']:8d} {a['rec']:9d} {a['err']:5d} "
          f"{b:10.3f} {af:9.3f} {af - b:+6.3f}")
  print(f"\n원본 백업: {BACKUP}")


if __name__ == "__main__":
  main()
