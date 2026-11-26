"""
변호사 검수 통과(lawyer-pass) 244건 서브셋에 대해 claude-opus-5 단독으로
12조건 전수를 돌린다. gpt-5.4-nano/claude-haiku-4-5 실행(run_two_models.py)과
동일한 데이터·조건·thinking=disabled 설정. 이메일은 시작/완료 2통만 보낸다.
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_api import run_condition, send_email  # noqa: E402
from prompt_builder import ALL_CONDITIONS, condition_key  # noqa: E402

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "data", "legal_dataset_lawyer_pass.jsonl")

MODEL = "claude-opus-5"
MODEL_SAFE = "claude-opus-5"


def build_html(results, title):
  html = f"<h2>{title}</h2>"
  html += ("<table border='1' cellpadding='5'><tr><th>ID</th><th>조건</th><th>Acc</th>"
           "<th>Macro-F1</th><th>거부</th><th>소요</th></tr>")
  for r in results:
    html += (f"<tr><td>{r['condition_id']}</td><td>{r['condition']}</td>"
             f"<td>{r['accuracy']:.3f}</td><td>{r['macro_f1']:.3f}</td>"
             f"<td>{r['refusals']}</td><td>{r['elapsed_min']}분</td></tr>")
  html += "</table>"
  return html


async def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--limit", type=int, default=244)
  ap.add_argument("--concurrency", type=int, default=8)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--fresh", action="store_true")
  args = ap.parse_args()

  with open(DATA_PATH, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

  send_email(
    "[KLDBench] 실험 시작: lawyer-pass 244건 x claude-opus-5",
    f"<p>변호사 검수 통과 {len(rows)}건 x {len(ALL_CONDITIONS)}조건, 모델: {MODEL}</p>",
  )

  import anthropic
  client = anthropic.AsyncAnthropic()

  results = []
  t0 = time.time()
  for ci, (hr, sl, lg) in enumerate(ALL_CONDITIONS, 1):
    r = await run_condition(rows, MODEL, MODEL_SAFE, "claude", client,
                             hr, sl, lg, args.limit, args.concurrency, args.max_tokens,
                             "disabled", args.fresh, ci, len(ALL_CONDITIONS))
    results.append(r)

  send_email(
    "[KLDBench] 실험 완료: lawyer-pass 244건 x claude-opus-5",
    build_html(results, f"claude-opus-5 완료 ({round((time.time()-t0)/60,1)}분 소요)"),
  )
  print("\n전체 완료", flush=True)


if __name__ == "__main__":
  asyncio.run(main())
