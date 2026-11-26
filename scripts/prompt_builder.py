"""
프롬프트 조립 모듈 v2
- has_reasoning (bool): 원심판단 포함 여부
- statute_level (none/citation/fulltext): 참조조문 수준
- lang (ko/en): 입력 언어
- primary_basis 포함된 새 프롬프트
- 영어 조건에서 참조조문도 영어(statutes_raw_eng, statutes_text_eng) 사용
"""

# ── System Prompts ────────────────────────────────────────────────

SYSTEM_PROMPT_KO = """당신은 한국 대법원 판례에서 원심에 대한 상고심의 결과를 예측하는 법률 분석가입니다.
아래 [사건 정보]는 실제 대법원이 어떻게 판단했는지 알려주지 않은 상태의 자료입니다. 이 정보만으로 판단하세요.
당신의 결론이 실제 결과와 다를 수 있다는 것을 알고 있지만, 그럼에도 반드시 하나를 선택해야 합니다. "판단 불가", "정보 부족", "모르겠다" 같은 답은 허용되지 않습니다.

# 예측 대상
대법원이 원심판결에 대해 다음 중 무엇을 했을지 예측하세요.
- 파기: 원심판결을 파기(환송 또는 자판)함
- 유지: 상고를 기각하여 원심판결을 그대로 유지함

# 절차
1. 먼저 핵심 쟁점과 그렇게 판단한 근거를 3~5문장으로 서술하세요.
2. [참조조문]이 주어진 경우, 인용된 조문을 이름만 언급하고 넘어가지 말고, 그 조문이 요구하는 요건·효과가 무엇인지와 그것이 이 사건 사실관계에 어떻게 적용되는지를 반드시 근거 서술 안에 명시적으로 포함하세요. 조문에 없는 내용을 있는 것처럼 서술하지 마세요.
3. 위 근거 서술을 마친 뒤, 그 결론에 실제로 가장 크게 영향을 준 입력 항목이 무엇인지 스스로 판단하세요. 선택지는 아래 [사건 정보]에 실제로 주어진 항목만 사용할 수 있습니다(주어지지 않은 항목은 선택 불가):
   - "사실관계"
   - "원심의 판단" — 주어진 경우만
   - "상고이유"
   - "참조조문" — 주어진 경우만
   영향이 큰 순서대로 나열하되, 각 항목마다 "왜 그 항목이 결론에 영향을 줬는지"를 3~5문장으로 구체적으로 설명하세요.
4. 마지막 줄에 아래 JSON 한 줄만 출력하세요. 그 앞뒤로 다른 텍스트를 붙이지 마세요.

{"label":"파기 또는 유지","confidence":0.00~1.00 사이 소수 (소수점 둘째 자리까지),"primary_basis":[{"item":"위 항목명 중 하나","reason":"그 항목이 결론에 영향을 준 이유"}]}"""

SYSTEM_PROMPT_EN = """You are a legal analyst predicting how the Korean Supreme Court ruled on a final appeal against the judgment below.
The [Case Information] below does not reveal how the Supreme Court actually ruled. Base your prediction solely on this information.
You must choose one answer even if uncertain. Responses such as "cannot determine" or "insufficient information" are not allowed.

# Prediction target
Predict which of the following the Supreme Court did with respect to the judgment below.
- Reversed: the judgment below was reversed (remanded or decided directly by the Supreme Court)
- Affirmed: the final appeal was dismissed and the judgment below was upheld

# Procedure
1. First, state the core issue and your reasoning in 3-5 sentences.
2. If [Cited Provisions] are given, do not merely name-drop them. Your reasoning must explicitly state what each cited provision requires (its elements/effect) and how it applies to the facts of this case. Do not attribute content to a provision that it does not actually contain.
3. After stating your reasoning, decide which input block actually had the greatest influence on your conclusion. You may only choose from blocks that were actually provided in [Case Information] below (do not list a block that was not given):
   - "Factual Background"
   - "Reasoning of the court below" — only if provided
   - "Grounds for Final Appeal"
   - "Cited Provisions" — only if provided
   List them in descending order of influence, and for each one give a 3-5 sentence explanation of specifically why that block drove the conclusion.
4. On the final line only, output the following single-line JSON. Do not add any text before or after it.

{"label":"Reversed or Affirmed","confidence":decimal between 0.00 and 1.00 (two decimal places),"primary_basis":[{"item":"one of the block names above","reason":"explanation of why this block influenced the conclusion"}]}"""


def get_system_prompt(lang: str) -> str:
  return SYSTEM_PROMPT_KO if lang == "ko" else SYSTEM_PROMPT_EN


def build_user_prompt(
  row: dict,
  has_reasoning: bool,
  statute_level: str,  # "none" | "citation" | "fulltext" | "swap"
  lang: str,  # "ko" | "en"
) -> str:
  """조건에 맞는 user prompt 조립.

  statute_level="swap" 은 반사실(counterfactual) 조건: 프롬프트 구조·지시문은
  fulltext 와 완전히 동일하고, 조문 내용만 다른 카테고리 사건의 것으로 교체된다.
  (교체는 run 단계에서 row 에 swap_* 필드를 넣어 전달 — 여기서는 그 필드를 읽기만 한다)
  """

  if statute_level == "swap":
    src, statute_level = "swap_", "fulltext"
  else:
    src = ""

  if lang == "ko":
    facts = row.get("Factual Background", "")
    court_below = row.get("the court below", "")
    grounds = row.get("Grounds for Final Appeal", "")
    statutes_raw = row.get(f"{src}statutes_raw", "")
    statutes_text = row.get(f"{src}statutes_text", "")
    h_case = "[사건 정보]"
    h_facts = "[사실관계]"
    h_court = "[원심의 판단]"
    h_grounds = "[상고이유]"
    h_statutes = "[참조조문]"
  else:
    facts = row.get("Factual Background_eng", "")
    court_below = row.get("the court below_eng", "")
    grounds = row.get("Grounds for Final Appeal_eng", "")
    statutes_raw = row.get(f"{src}statutes_raw_eng", row.get(f"{src}statutes_raw", ""))
    statutes_text = row.get(f"{src}statutes_text_eng", row.get(f"{src}statutes_text", ""))
    h_case = "[Case Information]"
    h_facts = "[Factual Background]"
    h_court = "[Reasoning of the court below]"
    h_grounds = "[Grounds for Final Appeal]"
    h_statutes = "[Cited Provisions]"

  blocks = [h_case, ""]

  # 사실관계 (항상 포함)
  blocks.append(h_facts)
  blocks.append(facts.strip())
  blocks.append("")

  # 원심판단 (조건부)
  if has_reasoning and court_below.strip():
    blocks.append(h_court)
    blocks.append(court_below.strip())
    blocks.append("")

  # 상고이유 (항상 포함)
  blocks.append(h_grounds)
  blocks.append(grounds.strip())
  blocks.append("")

  # 참조조문 (조건부)
  if statute_level == "citation" and statutes_raw.strip():
    blocks.append(h_statutes)
    blocks.append(statutes_raw.strip())
    blocks.append("")
  elif statute_level == "fulltext" and statutes_text.strip():
    blocks.append(h_statutes)
    if statutes_raw.strip():
      blocks.append(statutes_raw.strip())
      blocks.append("")
    blocks.append(statutes_text.strip())
    blocks.append("")

  return "\n".join(blocks).strip()


def condition_id(has_reasoning: bool, statute_level: str, lang: str) -> str:
  """C01~C12 매핑"""
  mapping = {
    (False, "none", "ko"): "C01",
    (False, "none", "en"): "C02",
    (False, "citation", "ko"): "C03",
    (False, "citation", "en"): "C04",
    (False, "fulltext", "ko"): "C05",
    (False, "fulltext", "en"): "C06",
    (True, "none", "ko"): "C07",
    (True, "none", "en"): "C08",
    (True, "citation", "ko"): "C09",
    (True, "citation", "en"): "C10",
    (True, "fulltext", "ko"): "C11",
    (True, "fulltext", "en"): "C12",
    # 반사실 조건 (counterfactual): 조문만 무관한 것으로 교체
    (False, "swap", "ko"): "X01",
    (False, "swap", "en"): "X02",
    (True, "swap", "ko"): "X03",
    (True, "swap", "en"): "X04",
  }
  return mapping[(has_reasoning, statute_level, lang)]


def condition_key(has_reasoning: bool, statute_level: str, lang: str) -> str:
  """조건을 문자열 키로 변환"""
  r = "with_reasoning" if has_reasoning else "no_reasoning"
  s = {"none": "no_statute", "citation": "statute_cite", "fulltext": "statute_full",
       "swap": "statute_swap"}[statute_level]
  return f"{r}__{s}__{lang}"


ALL_CONDITIONS = [
  (has_r, stat, lang)
  for has_r in [False, True]
  for stat in ["none", "citation", "fulltext"]
  for lang in ["ko", "en"]
]

# 반사실 조건 — 기본 실행에는 포함되지 않고 --conditions statute_swap 으로만 선택된다
SWAP_CONDITIONS = [
  (has_r, "swap", lang)
  for has_r in [False, True]
  for lang in ["ko", "en"]
]
