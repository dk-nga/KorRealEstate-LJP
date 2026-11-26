# KorRealEstate-LJP

Code and data for **"A Controlled Study of LLMs on Korean Real-Estate Judgment Prediction"**
(Findings of the ACL, ARR 2026 August cycle).

Do-Kyung Kim, Joonghyuk Hahn, Wonseok Lee, Hyunjae Kim, Yo-Sub Han
(Yonsei University, Linköping University, NextGenAI, Kim & Chang).
Corresponding author: Yo-Sub Han (emmous@yonsei.ac.kr).

Each Korean Supreme Court real-estate decision is reconstructed into input blocks
(facts, grounds for final appeal, optionally the lower court's reasoning and the
listed reference provisions) with the Court's disposition withheld; models predict
`REVERSED` vs. `AFFIRMED`. A 2 × 3 × 2 design crosses lower-court reasoning,
statutory material (none / citations / citations + text) and Korean vs. released
machine-translated English input. A counterfactual arm replaces the provision block
with one from a case of another category.

## Contents

```
data/
  legal_dataset_v2_full.jsonl        1,147 records (Korean + English fields)
  legal_dataset_v2_translated.jsonl  released English renderings
  legal_dataset_lawyer_pass.jsonl    244 records passing the lawyer audit
  swap_map.json                      fixed-seed donor assignment for the swap arm
prompts/
  construction_ko.md             block-construction prompt (verbatim Korean)
  translation_case_blocks.md     Korean→English prompt for the case blocks (verbatim)
  translation_statutes.md        Korean→English prompt for the provisions (verbatim)
scripts/
  prompt_builder.py   condition → prompt (12 main + 4 swap conditions)
  output_parser.py    verdict / confidence parsing
  run_vllm.py         open-weight models served with vLLM
  run_api.py          API models (Anthropic / OpenAI)
  run_two_models.py, run_opus5.py   API runs on the 244 audited records
  make_swap.py        builds swap_map.json
  analyze*.py, make_figures.py, audit_paper_numbers.py   paper tables and figures
  make_fig_signature.py   Figure 2 (aggregate vs paired effects, all four interventions)
  analyze_opus_swap.py    statute swap for claude-opus-5 on the 244 audited records
run_0.sh, run_1.sh    one queue per GPU for the open-weight suite
```

## Reproducing

Open-weight models (vLLM, one queue per GPU):

```bash
python -m venv .venv && source .venv/bin/activate && pip install vllm requests
CUDA_VISIBLE_DEVICES=0 bash run_0.sh   # gpt-oss-20b, Midm-2.0-Base, EXAONE-3.5-32B
CUDA_VISIBLE_DEVICES=1 bash run_1.sh   # Qwen3.6-27B, Gemma-4-31B-it
CONDITIONS=statute_swap LOGTAG=_swap CUDA_VISIBLE_DEVICES=0 bash run_0.sh   # swap arm only
```

API models on the audited subset:

```bash
pip install anthropic openai
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
python scripts/run_api.py --provider claude --model claude-opus-5 \
  --data data/legal_dataset_lawyer_pass.jsonl --limit 244            # 12 main conditions
python scripts/run_api.py --provider claude --model claude-opus-5 \
  --data data/legal_dataset_lawyer_pass.jsonl --limit 244 --conditions swap   # 4 swap conditions
```

Outputs go to `results/<model>__<condition>.jsonl`, one line per case. Runs resume
from existing files unless `--fresh` is given. Then:

```bash
python scripts/analyze.py && python scripts/analyze_swap.py && python scripts/make_figures.py
python scripts/make_fig_signature.py && python scripts/analyze_opus_swap.py
```

Optional run notifications are sent only if `RESEND_API_KEY`, `NOTIFY_FROM` and
`NOTIFY_EMAILS` are set.

## Data and license

Source decisions are published Korean Supreme Court judgments retrieved from the
National Law Information Center (law.go.kr), distributed with party identifiers
redacted; court judgments are excluded from copyright protection under Article 7 of
the Korean Copyright Act. See the paper's Ethics Statement.

- **Data** (`data/`, `prompts/`): CC BY 4.0
- **Code** (`scripts/`, `run_*.sh`): MIT — see `LICENSE`

The English fields are machine translations (`gpt-5-mini`, prompts in `prompts/`)
and have not been fully reviewed by a legal expert.

## Citation

```bibtex
@inproceedings{kim-etal-2027-korean-realestate-ljp,
  title  = {A Controlled Study of {LLM}s on {K}orean Real-Estate Judgment Prediction},
  author = {Kim, Do-Kyung and Hahn, Joonghyuk and Lee, Wonseok and Kim, Hyunjae and Han, Yo-Sub},
  booktitle = {Findings of the Association for Computational Linguistics},
  year   = {2027}
}
```
