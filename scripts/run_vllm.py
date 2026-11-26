"""
vLLM 기반 판결예측 실험 v2
- primary_basis 파싱 포함
- <think> 태그 제거
- 모델 전체 완료 시에만 이메일 발송
- 영어 조건에서 조문도 영어 사용
"""

import json
import re
import time
import argparse
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from prompt_builder import (
  get_system_prompt, build_user_prompt, condition_key, condition_id,
  ALL_CONDITIONS, SWAP_CONDITIONS
)
from output_parser import parse_prediction

VLLM_URL = "http://localhost:{port}/v1/chat/completions"

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")  # optional: run notifications
FROM_EMAIL = os.environ.get("NOTIFY_FROM", "")
TO_EMAILS = [e for e in os.environ.get("NOTIFY_EMAILS", "").split(",") if e]

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "data", "legal_dataset_v2_full.jsonl")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")


def send_email(subject, html_body):
  if not (RESEND_API_KEY and FROM_EMAIL and TO_EMAILS):
    return
  try:
    import resend
    resend.api_key = RESEND_API_KEY
    resend.Emails.send({"from": FROM_EMAIL, "to": TO_EMAILS, "subject": subject, "html": html_body})
    print(f"  [email] {subject}", flush=True)
  except Exception as e:
    print(f"  [email failed] {e}", flush=True)


# 연속 통신 실패가 이 수를 넘으면 실행을 중단한다 (서버 다운을 조용한 전량 실패로 만들지 않기 위해)
ERROR_ABORT_THRESHOLD = 25


def call_vllm(model_name, system, user, port=8001, timeout=300, max_tokens=1024, no_think=False):
  payload = {
    "model": model_name,
    "messages": [
      {"role": "system", "content": system},
      {"role": "user", "content": user},
    ],
    "temperature": 0,
    "max_tokens": max_tokens,
    "seed": 42,
  }
  if no_think:
    # Qwen3 계열 hybrid-thinking 모델의 내장 사고 비활성화 (타 모델 템플릿은 무시)
    payload["chat_template_kwargs"] = {"enable_thinking": False}
  try:
    resp = requests.post(
      VLLM_URL.format(port=port),
      json=payload,
      timeout=timeout,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
      # reasoning 모델(gpt-oss 등)이 예산을 reasoning 채널에서 소진하면 content가 None
      content = (msg.get("reasoning_content") or "").strip()
    return content
  except Exception as e:
    return f"ERROR: {e}"


def write_status(model_safe, payload):
  """results/_status_<model>.json 갱신 — cat 한 번으로 진행 상황 확인용"""
  os.makedirs(RESULTS_DIR, exist_ok=True)
  path = os.path.join(RESULTS_DIR, f"_status_{model_safe}.json")
  payload["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
  os.replace(tmp, path)


def run_condition(rows, model_name, model_safe, has_reasoning, statute_level, lang, port, limit, concurrency=16, cond_idx=1, cond_count=1, max_tokens=2048, fresh=False, no_think=False):
  cond_key = condition_key(has_reasoning, statute_level, lang)
  cond_id = condition_id(has_reasoning, statute_level, lang)
  system_prompt = get_system_prompt(lang)
  target_rows = rows[:limit]
  total = len(target_rows)

  print(f"\n{'='*60}", flush=True)
  print(f"모델: {model_name} | {cond_id} ({cond_key}) | {total}건 | 동시 {concurrency}", flush=True)
  print(f"시작: {datetime.now().strftime('%H:%M:%S')}", flush=True)

  os.makedirs(RESULTS_DIR, exist_ok=True)
  out_path = f"{RESULTS_DIR}/{model_safe}__{cond_key}.jsonl"

  # resume: 기존 결과가 있으면 완료된 case_id는 건너뛰고 이어서 진행 (--fresh면 무시하고 처음부터)
  prev_results = []
  if not fresh and os.path.exists(out_path):
    with open(out_path, encoding="utf-8") as f:
      prev_results = [json.loads(l) for l in f if l.strip()]
    done_ids = {r["case_id"] for r in prev_results}
    target_rows = [r for r in target_rows if r["case_id"] not in done_ids]
    if prev_results:
      print(f"  이어하기: {len(prev_results)}건 완료됨, {len(target_rows)}건 남음", flush=True)

  results = list(prev_results)
  correct = sum(1 for r in prev_results if r["correct"])
  total_done = len(prev_results)
  new_done = 0
  t_start = time.time()

  # 건별 즉시 저장 (중단 시에도 완료분 보존)
  out_f = open(out_path, "a" if prev_results else "w", encoding="utf-8")

  def do_one(row):
    user_prompt = build_user_prompt(row, has_reasoning, statute_level, lang)
    t0 = time.time()
    mt = max_tokens
    raw_output = ""
    attempt = 0
    for attempt in range(1, 4):
      raw_output = call_vllm(model_name, system_prompt, user_prompt, port, max_tokens=mt, no_think=no_think)
      if raw_output.startswith("ERROR:"):
        time.sleep(3 * attempt)  # 일시 오류: 같은 설정으로 재시도
        continue
      if parse_prediction(raw_output, lang)["predicted_label"] in ("파기", "유지"):
        break
      mt = min(mt * 2, 8192)  # label 없음 = 잘림 의심: 출력 예산 2배로 재시도 (temp=0이라 동일 재시도는 무의미)
    return raw_output, time.time() - t0, attempt

  with ThreadPoolExecutor(max_workers=concurrency) as pool:
    outputs = pool.map(do_one, target_rows)  # 입력 순서 보존, 내부적으로 동시 실행

    consecutive_errors = 0
    for i, (row, (raw_output, elapsed_call, attempts)) in enumerate(zip(target_rows, outputs)):
      # 3회 재시도가 전부 통신 오류로 끝난 건은 **기록하지 않는다**. 기록하면 resume 이
      # "완료된 case_id" 로 보고 영구히 건너뛰어, 서버가 죽은 구간이 빈 라벨로 굳는다.
      # (실제로 발생: 포트 충돌 1,147건 + 서버 다운 275건)
      if raw_output.startswith("ERROR:"):
        consecutive_errors += 1
        if consecutive_errors >= ERROR_ABORT_THRESHOLD:
          out_f.close()
          raise SystemExit(
            f"[abort] 연속 {consecutive_errors}건 통신 실패 — 서버가 죽은 것으로 판단.\n"
            f"        마지막 오류: {raw_output[:160]}\n"
            f"        기록된 {total_done}건은 유효하며, 재실행하면 나머지부터 이어간다.")
        continue
      consecutive_errors = 0

      parsed = parse_prediction(raw_output, lang)
      golden = row["label"]
      is_correct = parsed["predicted_label"] == golden
      if is_correct:
        correct += 1
      total_done += 1
      new_done += 1

      results.append({
        "case_id": row["case_id"],
        "condition_id": cond_id,
        "has_reasoning": has_reasoning,
        "statute_level": statute_level,
        "lang": lang,
        "golden_label": golden,
        "predicted_label": parsed["predicted_label"],
        "confidence": parsed["confidence"],
        "correct": is_correct,
        "reasoning_text": parsed["reasoning_text"],
        "primary_basis_json": parsed["primary_basis_json"],
        "primary_basis_top_item": parsed["primary_basis_top_item"],
        "primary_basis_top_reason": parsed["primary_basis_top_reason"],
        "elapsed_sec": round(elapsed_call, 2),
        "attempts": attempts,
      })
      out_f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
      out_f.flush()

      # 진행 상황: 10건마다 status 파일 갱신
      elapsed = time.time() - t_start
      eta = (total - total_done) / (new_done / elapsed) if new_done > 0 else 0
      if total_done % 10 == 0 or total_done == total:
        write_status(model_safe, {
          "model": model_name,
          "condition": f"{cond_id} ({cond_key})",
          "condition_progress": f"{cond_idx}/{cond_count}",
          "done": total_done,
          "total": total,
          "acc_so_far": round(correct / total_done, 3),
          "eta_min": round(eta / 60, 1),
        })
      if total_done % 50 == 0 or total_done == total:
        acc = correct / total_done
        print(f"  [{cond_idx}/{cond_count}] {cond_id} | {total_done}/{total} | acc={acc:.3f} | ETA {eta/60:.1f}분", flush=True)

  out_f.close()
  acc = correct / total_done if total_done > 0 else 0

  # F1 계산
  tp = sum(1 for r in results if r["predicted_label"] == "파기" and r["golden_label"] == "파기")
  fp = sum(1 for r in results if r["predicted_label"] == "파기" and r["golden_label"] == "유지")
  fn = sum(1 for r in results if r["predicted_label"] == "유지" and r["golden_label"] == "파기")
  tn = sum(1 for r in results if r["predicted_label"] == "유지" and r["golden_label"] == "유지")
  p_r = tp / (tp + fp) if (tp + fp) > 0 else 0
  r_r = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1_r = 2 * p_r * r_r / (p_r + r_r) if (p_r + r_r) > 0 else 0
  # 유지 F1
  p_a = tn / (tn + fn) if (tn + fn) > 0 else 0
  r_a = tn / (tn + fp) if (tn + fp) > 0 else 0
  f1_a = 2 * p_a * r_a / (p_a + r_a) if (p_a + r_a) > 0 else 0
  macro_f1 = (f1_r + f1_a) / 2

  elapsed_total = time.time() - t_start
  print(f"  완료: acc={acc:.3f} macro_f1={macro_f1:.3f} F1(파기)={f1_r:.3f} F1(유지)={f1_a:.3f} | {elapsed_total/60:.1f}분", flush=True)

  return {
    "condition_id": cond_id,
    "condition": cond_key,
    "accuracy": acc,
    "macro_f1": macro_f1,
    "f1_reversed": f1_r,
    "f1_affirmed": f1_a,
    "total": total_done,
    "correct": correct,
    "elapsed_min": round(elapsed_total / 60, 1),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True)
  parser.add_argument("--model-safe", default=None)
  parser.add_argument("--port", type=int, default=8001)
  parser.add_argument("--data", default=DATA_PATH)
  parser.add_argument("--limit", type=int, default=1147)
  parser.add_argument("--concurrency", type=int, default=16)
  parser.add_argument("--max-tokens", type=int, default=2048)
  parser.add_argument("--fresh", action="store_true", help="기존 결과 무시하고 처음부터")
  parser.add_argument("--no-think", action="store_true", help="hybrid-thinking 모델의 내장 사고 비활성화 (Qwen3 계열)")
  parser.add_argument("--conditions", default="all")
  args = parser.parse_args()

  model_safe = args.model_safe or args.model.replace("/", "_").replace(":", "_")

  with open(args.data, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

  # 반사실 조건이 요청되면 각 사건에 기증자 사건의 조문을 swap_* 필드로 주입
  if "swap" in args.conditions:
    swap_path = os.path.join(_BASE_DIR, "data", "swap_map.json")
    with open(swap_path, encoding="utf-8") as f:
      swap_map = json.load(f)
    idx = {r["case_id"]: r for r in rows}
    for r in rows:
      d = idx[swap_map[r["case_id"]]]
      r["swap_donor_id"] = d["case_id"]
      for k in ["statutes_raw", "statutes_text", "statutes_raw_eng", "statutes_text_eng"]:
        r[f"swap_{k}"] = d.get(k, "")
    print(f"반사실 조문 주입 완료 (기증자: 다른 카테고리 사건, data/swap_map.json)", flush=True)

  # ── preflight: 이 포트가 실제로 이 모델을 서빙하는가 ─────────────────
  # 같은 포트에 두 서버를 띄우면 늦게 뜬 쪽은 바인딩에 실패하고, 러너는 옆 GPU 의
  # 다른 모델에게 요청을 보내 404 만 받는다. 그대로 두면 1,147건 전부 빈 라벨로
  # 조용히 채워지므로 (실제로 발생함), 시작 전에 반드시 확인하고 크게 실패한다.
  try:
    served = [m["id"] for m in
              requests.get(f"http://localhost:{args.port}/v1/models", timeout=30).json()["data"]]
  except Exception as e:
    raise SystemExit(f"[preflight] 포트 {args.port} 조회 실패: {e}")
  if args.model not in served:
    raise SystemExit(
      f"[preflight] 포트 {args.port} 는 {served} 를 서빙 중 — 요청 모델 {args.model} 과 불일치.\n"
      f"            포트 충돌로 보인다. PORT 환경변수로 서로 다른 포트를 지정할 것.")
  print(f"[preflight] OK — 포트 {args.port} 에서 {args.model} 확인", flush=True)

  print(f"데이터: {len(rows)}건 | 모델: {args.model} | limit: {args.limit}", flush=True)

  pool = ALL_CONDITIONS + SWAP_CONDITIONS if "swap" in args.conditions else ALL_CONDITIONS
  conditions = pool
  if args.conditions != "all":
    filters = set(args.conditions.split(","))
    conditions = [
      (hr, sl, lg) for hr, sl, lg in pool
      if all(f in condition_key(hr, sl, lg) for f in filters)
    ]

  print(f"조건: {len(conditions)}개", flush=True)

  all_results = []
  half = len(conditions) // 2
  for ci, (hr, sl, lg) in enumerate(conditions, 1):
    result = run_condition(
      rows, args.model, model_safe, hr, sl, lg, args.port, args.limit,
      args.concurrency, cond_idx=ci, cond_count=len(conditions),
      max_tokens=args.max_tokens, fresh=args.fresh, no_think=args.no_think,
    )
    all_results.append(result)
    # 50% 중간 메일
    if ci == half:
      mid_html = "<table border='1' cellpadding='3'><tr><th>ID</th><th>조건</th><th>Acc</th><th>Macro-F1</th></tr>"
      for r in all_results:
        mid_html += f"<tr><td>{r['condition_id']}</td><td>{r['condition']}</td><td>{r['accuracy']:.3f}</td><td>{r['macro_f1']:.3f}</td></tr>"
      mid_html += "</table>"
      send_email(
        f"[KLDBench] 중간: {model_safe} ({ci}/{len(conditions)} 조건 완료)",
        f"<h3>{model_safe} 중간 진행 ({ci}/{len(conditions)})</h3>{mid_html}",
      )

  # 전체 요약 메일
  summary_html = "<table border='1' cellpadding='5'><tr><th>ID</th><th>조건</th><th>Acc</th><th>Macro-F1</th><th>F1(파기)</th><th>F1(유지)</th><th>소요</th></tr>"
  for r in all_results:
    summary_html += f"<tr><td>{r['condition_id']}</td><td>{r['condition']}</td><td>{r['accuracy']:.3f}</td><td>{r['macro_f1']:.3f}</td><td>{r['f1_reversed']:.3f}</td><td>{r['f1_affirmed']:.3f}</td><td>{r['elapsed_min']}분</td></tr>"
  summary_html += "</table>"

  send_email(
    f"[KLDBench] 완료: {model_safe} ({len(all_results)}개 조건)",
    f"<h2>판결예측 실험 완료: {model_safe}</h2>{summary_html}",
  )

  print(f"\n{'='*60}", flush=True)
  print(f"전체 완료: {model_safe} | {len(all_results)}개 조건", flush=True)
  for r in all_results:
    print(f"  {r['condition_id']} {r['condition']}: acc={r['accuracy']:.3f} mf1={r['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
  main()
