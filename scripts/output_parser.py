"""
모델 출력 파싱 — run_vllm.py 와 reparse_results.py 가 공유하는 단일 구현.

핵심: JSON 이 한 줄에 있다고 가정하지 말 것. 모델(midm 등)은 reason 필드가 길면
JSON 을 여러 줄에 걸쳐 출력한다. 중괄호 균형을 세어 마지막 완결 객체를 찾는다.
"""

import json
import re


def _iter_json_objects(text):
  """text 안의 균형 잡힌 {...} 객체를 (start, end, substring) 으로 순회 (문자열/이스케이프 인식)."""
  depth = 0
  start = None
  in_str = False
  esc = False
  for i, ch in enumerate(text):
    if in_str:
      if esc:
        esc = False
      elif ch == "\\":
        esc = True
      elif ch == '"':
        in_str = False
      continue
    if ch == '"':
      in_str = True
    elif ch == "{":
      if depth == 0:
        start = i
      depth += 1
    elif ch == "}":
      if depth > 0:
        depth -= 1
        if depth == 0 and start is not None:
          yield start, i + 1, text[start:i + 1]
          start = None


def _normalize_label(raw_label, lang):
  raw = str(raw_label)
  if lang == "ko":
    if "파기" in raw:
      return "파기"
    if "유지" in raw:
      return "유지"
    return raw
  low = raw.lower()
  if "reversed" in low or "파기" in raw:
    return "파기"
  if "affirmed" in low or "유지" in raw:
    return "유지"
  return raw


def parse_prediction(text, lang):
  """모델 출력에서 label / confidence / primary_basis / reasoning 추출."""
  # 사고 블록 제거 — <think>(로컬 모델) / <thinking>(사고 비활성화 시 Claude 가 흘리는 경우)
  text_clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL).strip()
  if not text_clean:
    text_clean = text

  prediction = {}
  json_start = None

  # 1) 균형 잡힌 JSON 객체 중 "label" 키를 가진 마지막 것 (여러 줄 걸쳐 있어도 잡힘)
  for s, _e, chunk in _iter_json_objects(text_clean):
    if '"label"' not in chunk:
      continue
    try:
      obj = json.loads(chunk)
    except json.JSONDecodeError:
      # 흔한 깨짐 보정: 후행 콤마, 스마트 따옴표
      fixed = re.sub(r",\s*([}\]])", r"\1", chunk).replace("“", '"').replace("”", '"')
      try:
        obj = json.loads(fixed)
      except json.JSONDecodeError:
        continue
    if isinstance(obj, dict) and "label" in obj:
      prediction, json_start = obj, s

  # 2) 완결 JSON 이 없으면(잘림 등) label/confidence 만이라도 정규식으로 건짐
  if not prediction:
    m = None
    for m in re.finditer(r'"label"\s*:\s*"([^"]*)"', text_clean, re.DOTALL):
      pass  # 마지막 매치 사용
    if m:
      prediction = {"label": m.group(1)}
      brace = text_clean.rfind("{", 0, m.start())
      json_start = brace if brace != -1 else m.start()
      mc = None
      for mc in re.finditer(r'"confidence"\s*:\s*([0-9.]+)', text_clean):
        pass
      if mc:
        prediction["confidence"] = mc.group(1)

  reasoning_text = (text_clean[:json_start] if json_start is not None else text_clean).strip()
  normalized = _normalize_label(prediction.get("label", ""), lang)

  # primary_basis — 모델이 dict 배열 대신 문자열 배열/단일 값을 줄 수 있음
  pb = prediction.get("primary_basis", [])
  if not isinstance(pb, list):
    pb = [pb] if pb else []
  first = pb[0] if pb else None
  if isinstance(first, dict):
    pb_top_item = str(first.get("item", ""))
    pb_top_reason = str(first.get("reason", ""))
  else:
    pb_top_item = str(first) if first is not None else ""
    pb_top_reason = ""

  try:
    confidence = float(prediction.get("confidence", 0.0))
  except (TypeError, ValueError):
    confidence = 0.0

  return {
    "reasoning_text": reasoning_text,
    "predicted_label": normalized,
    "confidence": confidence,
    "primary_basis_json": json.dumps(pb, ensure_ascii=False) if pb else "[]",
    "primary_basis_top_item": pb_top_item,
    "primary_basis_top_reason": pb_top_reason,
  }
