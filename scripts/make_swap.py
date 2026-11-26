"""
반사실(counterfactual) 조문 배정 생성.

각 사건에 '다른 카테고리' 사건의 조문을 기증자(donor)로 배정한다. 완전 무관한
조문을 진짜인 것처럼 제시했을 때 예측이 바뀌는지를 보기 위한 것이므로, 같은
카테고리 내 교체(우연히 관련될 수 있음)는 쓰지 않는다.

배정은 고정 시드로 결정적이며, 결과는 data/swap_map.json 에 저장해 재현 가능하게 한다.
"""

import json
import os
import random
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "legal_dataset_v2_full.jsonl")
OUT = os.path.join(BASE, "data", "swap_map.json")
SEED = 42


def main():
  rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
  by_cat = defaultdict(list)
  for r in rows:
    by_cat[r["category"]].append(r["case_id"])
  cats = sorted(by_cat)

  rng = random.Random(SEED)
  mapping = {}
  for r in sorted(rows, key=lambda x: x["case_id"]):
    others = [c for c in cats if c != r["category"] and by_cat[c]]
    donor_cat = rng.choice(others)
    mapping[r["case_id"]] = rng.choice(sorted(by_cat[donor_cat]))

  with open(OUT, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=0, sort_keys=True)

  # 검증
  idx = {r["case_id"]: r for r in rows}
  assert all(idx[k]["category"] != idx[v]["category"] for k, v in mapping.items()), \
      "same-category donor found"
  assert all(k != v for k, v in mapping.items()), "self-donor found"
  overlap = sum(1 for k, v in mapping.items()
                if set(_arts(idx[k])) & set(_arts(idx[v])))
  print(f"배정 {len(mapping)}건 → {OUT}")
  print(f"  자기 자신 배정: 0 | 같은 카테고리 배정: 0 (검증 통과)")
  print(f"  기증자와 조문 번호가 하나라도 겹치는 사건: {overlap} ({overlap/len(mapping):.1%})")
  print(f"  기증자 사용 분포: 고유 기증자 {len(set(mapping.values()))}건")


def _arts(r):
  import re
  return re.findall(r"[가-힣]+법[^,/]*?제[0-9]+조", r["statutes_raw"])


if __name__ == "__main__":
  main()
