"""
API 모델(Claude, GPT) 판결예측 실험.

run_vllm.py 와 동일한 결과 스키마·재개(resume)·재시도 정책을 쓴다 — 로컬 5모델과
같은 분석 스크립트로 처리할 수 있어야 하므로 스키마 일치가 필수다.

로컬 실험과의 대응
- 로컬: temperature=0, seed=42, 확장 사고 없음(qwen 은 enable_thinking=false)
- Claude: temperature/top_p/top_k 는 최신 모델에서 400 → 보내지 않는다.
          확장 사고는 기본 ON 이므로 --thinking disabled 로 끄는 것이 로컬과 가장 가깝다.
- GPT: reasoning 계열은 temperature 미지원 → max_completion_tokens 만 지정.

두 API 모두 seed 를 보장하지 않으므로 완전 재현은 불가 — 논문 각주에 명시할 것.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime

from prompt_builder import (
  get_system_prompt, build_user_prompt, condition_key, condition_id,
  ALL_CONDITIONS, SWAP_CONDITIONS
)
from output_parser import parse_prediction

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")  # optional: run notifications
FROM_EMAIL = os.environ.get("NOTIFY_FROM", "")
TO_EMAILS = [e for e in os.environ.get("NOTIFY_EMAILS", "").split(",") if e]

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "data", "legal_dataset_v2_full.jsonl")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
VALID = ("파기", "유지")


def send_email(subject, html_body):
  if not (RESEND_API_KEY and FROM_EMAIL and TO_EMAILS):
    return
  """resend 패키지가 로컬 python3.14(pyexpat 심볼 문제)에서 import 자체가 깨져서
  urllib로 Resend REST API를 직접 호출한다 — 의존성 없이 표준 라이브러리만 사용."""
  try:
    import urllib.request
    body = json.dumps({
      "from": FROM_EMAIL, "to": TO_EMAILS, "subject": subject, "html": html_body,
    }).encode("utf-8")
    req = urllib.request.Request(
      "https://api.resend.com/emails", data=body, method="POST",
      headers={
        "Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json",
        # Cloudflare가 Python-urllib 기본 UA를 봇으로 차단(403) — 브라우저 UA로 우회
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
      },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
      resp.read()
    print(f"  [email] {subject}", flush=True)
  except Exception as e:
    print(f"  [email failed] {e}", flush=True)


async def call_claude(client, model, system, user, max_tokens, thinking):
  """반환: (text, stop_reason). 거부(refusal)는 예외가 아니라 정상 응답으로 온다."""
  kwargs = {
    "model": model,
    "max_tokens": max_tokens,
    "system": system,
    "messages": [{"role": "user", "content": user}],
  }
  if thinking == "disabled":
    # 최신 모델은 사고가 기본 ON — 로컬 실험과 맞추려면 명시적으로 꺼야 한다
    kwargs["thinking"] = {"type": "disabled"}
  # temperature/top_p/top_k 는 보내지 않는다 (최신 Claude 에서 400)

  msg = await client.messages.create(**kwargs)
  if msg.stop_reason == "refusal":
    cat = getattr(getattr(msg, "stop_details", None), "category", None)
    return f"REFUSAL: {cat}", "refusal"
  text = "".join(b.text for b in msg.content if b.type == "text")
  return text.strip(), msg.stop_reason


async def call_openai(client, model, system, user, max_tokens, thinking):
  resp = await client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": system},
      {"role": "user", "content": user},
    ],
    max_completion_tokens=max_tokens,
  )
  choice = resp.choices[0]
  return (choice.message.content or "").strip(), choice.finish_reason


async def one_case(client, provider, model, system, user, lang, sem, max_tokens, thinking):
  """실패 원인별 재시도: 잘림 의심이면 예산 2배, 일시 오류면 백오프.

  SDK 예외는 종류를 가리지 않고 같은 정책으로 처리한다 — 두 SDK 의 예외 계층이
  다르고, 어차피 백오프 후 재시도가 유일한 대응이기 때문.
  """
  mt, attempt = max_tokens, 0
  text, stop = "", ""
  async with sem:
    for attempt in range(1, 4):
      try:
        if provider == "claude":
          text, stop = await call_claude(client, model, system, user, mt, thinking)
        else:
          text, stop = await call_openai(client, model, system, user, mt, thinking)
      except Exception as e:
        text, stop = f"ERROR: {type(e).__name__}: {e}", "error"
        await asyncio.sleep(5 * attempt)
        continue
      if stop == "refusal":
        break  # 재시도해도 동일 — 거부 자체가 측정값
      if parse_prediction(text, lang)["predicted_label"] in VALID:
        break
      mt = min(mt * 2, 8192)
  return text, stop, attempt


def write_status(model_safe, payload):
  os.makedirs(RESULTS_DIR, exist_ok=True)
  path = os.path.join(RESULTS_DIR, f"_status_{model_safe}.json")
  payload["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)


async def run_condition(rows, model, model_safe, provider, client, hr, sl, lg,
                        limit, concurrency, max_tokens, thinking, fresh,
                        cond_idx, cond_count):
  cond_key = condition_key(hr, sl, lg)
  cond_id = condition_id(hr, sl, lg)
  system_prompt = get_system_prompt(lg)
  target = rows[:limit]
  total = len(target)

  print(f"\n{'='*60}\n모델: {model} | {cond_id} ({cond_key}) | {total}건 | 동시 {concurrency}", flush=True)

  os.makedirs(RESULTS_DIR, exist_ok=True)
  out_path = f"{RESULTS_DIR}/{model_safe}__{cond_key}.jsonl"

  prev = []
  if not fresh and os.path.exists(out_path):
    with open(out_path, encoding="utf-8") as f:
      prev = [json.loads(l) for l in f if l.strip()]
    done = {r["case_id"] for r in prev}
    target = [r for r in target if r["case_id"] not in done]
    if prev:
      print(f"  이어하기: {len(prev)}건 완료됨, {len(target)}건 남음", flush=True)

  results = list(prev)
  correct = sum(1 for r in prev if r["correct"])
  total_done, new_done = len(prev), 0
  t0 = time.time()
  out_f = open(out_path, "a" if prev else "w", encoding="utf-8")
  sem = asyncio.Semaphore(concurrency)

  for start in range(0, len(target), concurrency * 2):
    batch = target[start:start + concurrency * 2]
    coros = [
      one_case(client, provider, model, system_prompt,
               build_user_prompt(row, hr, sl, lg), lg, sem, max_tokens, thinking)
      for row in batch
    ]
    t_batch = time.time()
    outs = await asyncio.gather(*coros)

    for row, (text, stop, attempts) in zip(batch, outs):
      p = parse_prediction(text, lg)
      ok = p["predicted_label"] == row["label"]
      correct += ok
      total_done += 1
      new_done += 1
      rec = {
        "case_id": row["case_id"],
        "condition_id": cond_id,
        "has_reasoning": hr,
        "statute_level": sl,
        "lang": lg,
        "golden_label": row["label"],
        "predicted_label": p["predicted_label"],
        "confidence": p["confidence"],
        "correct": ok,
        "format_ok": p["predicted_label"] in VALID and attempts == 1,
        "recovered": p["predicted_label"] in VALID and attempts > 1,
        "reasoning_text": p["reasoning_text"],
        "primary_basis_json": p["primary_basis_json"],
        "primary_basis_top_item": p["primary_basis_top_item"],
        "primary_basis_top_reason": p["primary_basis_top_reason"],
        "elapsed_sec": round((time.time() - t_batch) / max(len(batch), 1), 2),
        "attempts": attempts,
        "stop_reason": stop,
      }
      results.append(rec)
      out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_f.flush()

    el = time.time() - t0
    eta = (total - total_done) / (new_done / el) if new_done else 0
    write_status(model_safe, {
      "model": model, "condition": f"{cond_id} ({cond_key})",
      "condition_progress": f"{cond_idx}/{cond_count}",
      "done": total_done, "total": total,
      "acc_so_far": round(correct / total_done, 3) if total_done else 0,
      "eta_min": round(eta / 60, 1),
    })
    print(f"  [{cond_idx}/{cond_count}] {cond_id} | {total_done}/{total} | "
          f"acc={correct/total_done:.3f} | ETA {eta/60:.1f}분", flush=True)

  out_f.close()
  acc = correct / total_done if total_done else 0
  tp = sum(1 for r in results if r["predicted_label"] == "파기" and r["golden_label"] == "파기")
  fp = sum(1 for r in results if r["predicted_label"] == "파기" and r["golden_label"] == "유지")
  fn = sum(1 for r in results if r["predicted_label"] == "유지" and r["golden_label"] == "파기")
  tn = sum(1 for r in results if r["predicted_label"] == "유지" and r["golden_label"] == "유지")
  f1r = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
  f1a = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0
  refusals = sum(1 for r in results if r.get("stop_reason") == "refusal")
  print(f"  완료: acc={acc:.3f} macro_f1={(f1r+f1a)/2:.3f} 거부={refusals} | "
        f"{(time.time()-t0)/60:.1f}분", flush=True)

  return {"condition_id": cond_id, "condition": cond_key, "accuracy": acc,
          "macro_f1": (f1r + f1a) / 2, "f1_reversed": f1r, "f1_affirmed": f1a,
          "refusals": refusals, "elapsed_min": round((time.time() - t0) / 60, 1)}


async def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--provider", required=True, choices=["claude", "openai"])
  ap.add_argument("--model", required=True, help="예: claude-opus-5")
  ap.add_argument("--model-safe", default=None)
  ap.add_argument("--data", default=DATA_PATH)
  ap.add_argument("--limit", type=int, default=1147)
  ap.add_argument("--concurrency", type=int, default=8)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--thinking", default="disabled", choices=["disabled", "adaptive"],
                  help="claude 전용. 로컬 실험과 맞추려면 disabled")
  ap.add_argument("--fresh", action="store_true")
  ap.add_argument("--conditions", default="all")
  args = ap.parse_args()

  model_safe = args.model_safe or args.model.replace("/", "_").replace(":", "_")

  with open(args.data, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

  if "swap" in args.conditions:
    with open(os.path.join(_BASE_DIR, "data", "swap_map.json"), encoding="utf-8") as f:
      swap_map = json.load(f)
    # donor 는 swap_map 을 만든 전체 1,147건에서 찾는다. 서브셋(예: 검수 244건)만
    # 넣어도 오픈모델 반사실 실험과 같은 donor 블록이 주입되도록.
    with open(os.path.join(_BASE_DIR, "data", "legal_dataset_v2_full.jsonl"), encoding="utf-8") as f:
      idx = {d["case_id"]: d for d in map(json.loads, f)}
    for r in rows:
      d = idx[swap_map[r["case_id"]]]
      r["swap_donor_id"] = d["case_id"]
      for k in ["statutes_raw", "statutes_text", "statutes_raw_eng", "statutes_text_eng"]:
        r[f"swap_{k}"] = d.get(k, "")
    print("반사실 조문 주입 완료", flush=True)

  if args.provider == "claude":
    import anthropic
    client = anthropic.AsyncAnthropic()
  else:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

  pool = ALL_CONDITIONS + SWAP_CONDITIONS if "swap" in args.conditions else ALL_CONDITIONS
  conditions = pool
  if args.conditions != "all":
    filt = set(args.conditions.split(","))
    conditions = [c for c in pool if all(f in condition_key(*c) for f in filt)]

  print(f"데이터: {len(rows)}건 | 모델: {args.model} | 조건: {len(conditions)}개 | "
        f"thinking={args.thinking}", flush=True)

  all_results = []
  for ci, (hr, sl, lg) in enumerate(conditions, 1):
    r = await run_condition(rows, args.model, model_safe, args.provider, client,
                            hr, sl, lg, args.limit, args.concurrency,
                            args.max_tokens, args.thinking, args.fresh,
                            ci, len(conditions))
    all_results.append(r)

  html = ("<table border='1' cellpadding='5'><tr><th>ID</th><th>조건</th><th>Acc</th>"
          "<th>Macro-F1</th><th>거부</th><th>소요</th></tr>")
  for r in all_results:
    html += (f"<tr><td>{r['condition_id']}</td><td>{r['condition']}</td>"
             f"<td>{r['accuracy']:.3f}</td><td>{r['macro_f1']:.3f}</td>"
             f"<td>{r['refusals']}</td><td>{r['elapsed_min']}분</td></tr>")
  html += "</table>"
  send_email(f"[KLDBench] 완료: {model_safe} ({len(all_results)}개 조건)",
             f"<h2>판결예측 완료: {model_safe}</h2>{html}")

  print(f"\n전체 완료: {model_safe}", flush=True)
  for r in all_results:
    print(f"  {r['condition_id']} {r['condition']}: acc={r['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
  asyncio.run(main())
