"""
변호사 검수 통과(lawyer-pass) 244건 서브셋에 대해 gpt-5.4-nano / claude-haiku-4-5
두 모델을 12조건 전수로 돌린다.

run_api.py 의 run_condition()/send_email() 을 그대로 재사용하되, 이메일은
run_api.py처럼 모델당 1통이 아니라 전체 실행 기준 시작/중간/완료 3통만 보내도록
이 스크립트에서 별도로 제어한다.
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

RUNS = [
  {"provider": "openai", "model": "gpt-5.4-nano", "model_safe": "gpt-5.4-nano"},
  {"provider": "claude", "model": "claude-haiku-4-5-20251001", "model_safe": "claude-haiku-4-5"},
]


def build_html(all_summaries, title):
  html = f"<h2>{title}</h2>"
  for model_safe, results in all_summaries.items():
    html += f"<h3>{model_safe}</h3>"
    html += ("<table border='1' cellpadding='5'><tr><th>ID</th><th>조건</th><th>Acc</th>"
             "<th>Macro-F1</th><th>거부</th><th>소요</th></tr>")
    for r in results:
      html += (f"<tr><td>{r['condition_id']}</td><td>{r['condition']}</td>"
               f"<td>{r['accuracy']:.3f}</td><td>{r['macro_f1']:.3f}</td>"
               f"<td>{r['refusals']}</td><td>{r['elapsed_min']}분</td></tr>")
    html += "</table>"
  return html


async def run_model(cfg, rows, limit, concurrency, max_tokens, fresh):
  if cfg["provider"] == "claude":
    import anthropic
    client = anthropic.AsyncAnthropic()
  else:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

  results = []
  for ci, (hr, sl, lg) in enumerate(ALL_CONDITIONS, 1):
    r = await run_condition(rows, cfg["model"], cfg["model_safe"], cfg["provider"], client,
                             hr, sl, lg, limit, concurrency, max_tokens, "disabled", fresh,
                             ci, len(ALL_CONDITIONS))
    results.append(r)
  return results


async def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--limit", type=int, default=244)
  ap.add_argument("--concurrency", type=int, default=8)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--fresh", action="store_true")
  args = ap.parse_args()

  with open(DATA_PATH, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

  model_names = ", ".join(f"{r['model']} ({r['provider']})" for r in RUNS)
  send_email(
    "[KLDBench] 실험 시작: lawyer-pass 244건 x 2모델",
    f"<p>변호사 검수 통과 {len(rows)}건 x {len(ALL_CONDITIONS)}조건 x 2모델 시작</p>"
    f"<p>모델: {model_names}</p>",
  )

  all_summaries = {}
  t0 = time.time()
  for i, cfg in enumerate(RUNS):
    print(f"\n{'#'*60}\n모델 {i+1}/{len(RUNS)}: {cfg['model']}\n{'#'*60}", flush=True)
    results = await run_model(cfg, rows, args.limit, args.concurrency, args.max_tokens, args.fresh)
    all_summaries[cfg["model_safe"]] = results

    if i < len(RUNS) - 1:
      send_email(
        "[KLDBench] 중간 진행: 1/2 모델 완료",
        build_html(all_summaries, f"{cfg['model_safe']} 완료 ({round((time.time()-t0)/60,1)}분 경과)"),
      )

  send_email(
    "[KLDBench] 실험 완료: lawyer-pass 244건 x 2모델",
    build_html(all_summaries, f"전체 완료 ({round((time.time()-t0)/60,1)}분 소요)"),
  )
  print("\n전체 완료", flush=True)


if __name__ == "__main__":
  asyncio.run(main())
