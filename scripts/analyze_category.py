"""A2 (X4Pd W4): category-level paired effects of the four interventions.

For each subject category with >= 90 cases and each intervention, per case we average the
correctness difference over the matched condition pairs and over the five open-weight models
(the estimand of Section 4, pooled over models), and pool verdict changes over complete pairs.
Delta: mean per-case difference in %p with a case-bootstrap 95% CI (10,000 resamples, seed 42).
Flips: share of complete (model, pair) observations whose verdict changes.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
VALID = ("파기", "유지")
MODELS = ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]
HR, ST, LG = ["no_reasoning", "with_reasoning"], ["no_statute", "statute_cite", "statute_full"], ["ko", "en"]
PAIRS = {
  "reason": [(f"no_reasoning__{s}__{l}", f"with_reasoning__{s}__{l}") for s in ST for l in LG],
  "prov": [(f"{h}__no_statute__{l}", f"{h}__statute_full__{l}") for h in HR for l in LG],
  "swap": [(f"{h}__statute_full__{l}", f"{h}__statute_swap__{l}") for h in HR for l in LG],
  "lang": [(f"{h}__{s}__ko", f"{h}__{s}__en") for h in HR for s in ST],
}
CAT_EN = {"소유권·등기": "ownership \\& registration", "취득세": "acquisition tax", "양도소득세": "capital-gains tax",
          "임대차": "lease", "기타": "other", "재개발·재건축": "redevelopment"}


def load(m, c):
  return {r["case_id"]: r for r in (json.loads(l) for l in open(os.path.join(R, f"{m}__{c}.jsonl"), encoding="utf-8") if l.strip())}


def main():
  cat = {json.loads(l)["case_id"]: json.loads(l)["category"]
         for l in open(os.path.join(BASE, "data", "legal_dataset_v2_full.jsonl"), encoding="utf-8")}
  conds = sorted({c for p in PAIRS.values() for ab in p for c in ab})
  data = {m: {c: load(m, c) for c in conds} for m in MODELS}
  rng = np.random.default_rng(42)
  out = {}
  for k, pairs in PAIRS.items():
    per = defaultdict(lambda: [0.0, 0, 0, 0])  # case -> [sum d, n models, flips, complete]
    for m in MODELS:
      D = data[m]
      for i in set.intersection(*[set(D[a]) & set(D[b]) for a, b in pairs]):
        d = np.mean([D[b][i]["correct"] for _, b in pairs]) - np.mean([D[a][i]["correct"] for a, _ in pairs])
        f = n = 0
        for a, b in pairs:
          la, lb = D[a][i]["predicted_label"], D[b][i]["predicted_label"]
          if la in VALID and lb in VALID:
            n += 1; f += la != lb
        p = per[i]; p[0] += d; p[1] += 1; p[2] += f; p[3] += n
    for c in CAT_EN:
      ids = [i for i in per if cat.get(i) == c]
      d = np.array([per[i][0] / per[i][1] for i in ids]); f = np.array([per[i][2] for i in ids]); n = np.array([per[i][3] for i in ids])
      idx = rng.integers(0, len(ids), size=(10_000, len(ids)))
      ci = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
      out.setdefault(c, {})[k] = dict(n=len(ids), d=d.mean() * 100, ci=ci.tolist(), flip=f.sum() / n.sum() * 100)
  for c, v in out.items():
    print(f"{CAT_EN[c]:24s} n={v['reason']['n']:4d} " + " ".join(
      f"{k}: {v[k]['d']:+.1f} [{v[k]['ci'][0]:+.1f},{v[k]['ci'][1]:+.1f}] fl {v[k]['flip']:.1f}%" for k in PAIRS))
  json.dump(out, open(os.path.join(R, "_category_stats.json"), "w"), ensure_ascii=False, indent=1)
  # LaTeX rows
  for c, v in sorted(out.items(), key=lambda x: -x[1]["reason"]["n"]):
    cells = " & ".join(f"${v[k]['d']:+.1f}${'^{*}' if v[k]['ci'][0] > 0 or v[k]['ci'][1] < 0 else ''} & {v[k]['flip']:.1f}" for k in ["reason", "prov", "swap", "lang"])
    print(f"{CAT_EN[c]} & {v['reason']['n']} & {cells} \\\\")


if __name__ == "__main__":
  main()
