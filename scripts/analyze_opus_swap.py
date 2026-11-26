"""E1: counterfactual statute swap for claude-opus-5 on the 244 audited records.

Same estimands as Table swap / tab:direction: case-level accuracy difference (donor - target),
flip rate over complete pairs, net drift toward Reversed, exact sign test on per-case net direction.
Also reports the open-weight models restricted to the same 244 cases for comparison.
"""
import glob
import json
import os
from math import comb

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
VALID = ("파기", "유지")
PAIRS = [(f"{h}__statute_full__{l}", f"{h}__statute_swap__{l}")
         for h in ["no_reasoning", "with_reasoning"] for l in ["ko", "en"]]
MAIN = [f"{h}__{s}__{l}" for h in ["no_reasoning", "with_reasoning"]
        for s in ["no_statute", "statute_cite", "statute_full"] for l in ["ko", "en"]]


def load(model, cond):
  f = os.path.join(R, f"{model}__{cond}.jsonl")
  if not os.path.exists(f):
    return None
  return {r["case_id"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}


def sign_p(pos, neg):
  n = pos + neg
  k = min(pos, neg)
  return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def analyse(model, cases=None):
  data = {c: load(model, c) for a, b in PAIRS for c in (a, b)}
  if any(v is None for v in data.values()):
    return None
  ids = set.intersection(*[set(v) for v in data.values()])
  if cases is not None:
    ids &= cases
  ids = sorted(ids)
  d, fl, cp, net = [], [], [], []
  for i in ids:
    d.append(np.mean([data[b][i]["correct"] for _, b in PAIRS]) - np.mean([data[a][i]["correct"] for a, _ in PAIRS]))
    f = n = s = 0
    for a, b in PAIRS:
      la, lb = data[a][i]["predicted_label"], data[b][i]["predicted_label"]
      if la in VALID and lb in VALID:
        n += 1
        if la != lb:
          f += 1
          s += 1 if lb == "파기" else -1
    fl.append(f); cp.append(n); net.append(s)
  d, fl, cp, net = map(np.array, (d, fl, cp, net))
  rng = np.random.default_rng(42)
  idx = rng.integers(0, len(d), size=(10_000, len(d)))
  ci = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
  acc_t = np.mean([np.mean([data[a][i]["correct"] for a, _ in PAIRS]) for i in ids])
  acc_d = np.mean([np.mean([data[b][i]["correct"] for _, b in PAIRS]) for i in ids])
  pos, neg, zero = int((net > 0).sum()), int((net < 0).sum()), int((net == 0).sum())
  main_labels = [r["predicted_label"] for c in MAIN for r in (load(model, c) or {}).values()
                 if r["case_id"] in set(ids) and r["predicted_label"] in VALID]
  return dict(model=model, n=len(ids), acc_target=acc_t, acc_donor=acc_d, delta_pp=d.mean() * 100,
              ci=ci.tolist(), p=float(stats.ttest_1samp(d, 0).pvalue), flip=fl.sum() / cp.sum() * 100,
              complete_pairs=int(cp.sum()), drift_pp=net.sum() / (len(ids) * len(PAIRS)) * 100,
              pos=pos, neg=neg, zero=zero, p_dir=sign_p(pos, neg),
              p_rev_main=float(np.mean([l == "파기" for l in main_labels])) if main_labels else None)


if __name__ == "__main__":
  opus = analyse("claude-opus-5")
  out = {"opus": opus}
  if opus:
    audited = set(load("claude-opus-5", PAIRS[0][0]))
    out["open_on_244"] = [analyse(m, audited) for m in
                          ["gemma-4-31b-it", "qwen3.6-27b", "exaone-3.5-32b", "midm-2.0-base", "gpt-oss-20b"]]
  for r in [out.get("opus")] + out.get("open_on_244", []):
    if r:
      print(f"{r['model']:16s} n={r['n']} target={r['acc_target']:.3f} donor={r['acc_donor']:.3f} "
            f"d={r['delta_pp']:+.1f}%p CI[{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}] p={r['p']:.3f} "
            f"flip={r['flip']:.1f}% pairs={r['complete_pairs']} drift={r['drift_pp']:+.1f}%p "
            f"+/-/0={r['pos']}/{r['neg']}/{r['zero']} p_dir={r['p_dir']:.2g} P(Rev)={r['p_rev_main']}")
  if not opus:
    print("opus swap incomplete")
  json.dump(out, open(os.path.join(R, "_opus_swap_stats.json"), "w"), indent=1, ensure_ascii=False)
