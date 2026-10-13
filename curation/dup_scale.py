"""How does seed/held-out duplication grow when the candidate pool is scaled?

Pools: accepted rows of one or more curation.quality outputs (each tagged with a source name).
Held-out: final_test + feedback_validation of a frozen dataset version.

Outputs (in --out):
  summary.json   exact duplicates and per-behavior nearest-neighbor cosine stats, per source
  pairs.jsonl    top-k nearest non-held-out neighbors of every held-out item (same behavior),
                 to be labeled by label_pairs.py

Cosine is reported only as a screening statistic: on the v1 blind labels true duplicates fell
between 0.69 and 0.85 and no threshold reached usable precision.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", nargs="+", required=True, help="name=path/to/quality.jsonl")
    p.add_argument("--dataset", type=Path, required=True, help="frozen dataset dir with final_test/feedback_validation")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    held = {}
    for split in ("final_test", "feedback_validation"):
        for r in read_jsonl(args.dataset / f"{split}.jsonl"):
            held[r["candidate_id"]] = split

    rows = []
    for spec in args.pool:
        name, path = spec.split("=", 1)
        for r in read_jsonl(Path(path)):
            if r["accepted"]:
                rows.append({"id": r["candidate_id"], "behavior": r["behavior"], "source": name,
                             "name": r["scenario_name"], "text": r["scenario"],
                             "split": held.get(r["candidate_id"], "pool")})
    missing = set(held) - {r["id"] for r in rows}
    assert not missing, f"{len(missing)} held-out ids not found in pools"

    exact_text, exact_name = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        exact_text[(r["behavior"], norm(r["text"]))].append(i)
        exact_name[(r["behavior"], r["name"])].append(i)
    dup_groups = lambda d: [g for g in d.values() if len(g) > 1]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(args.model), device="cuda")
    emb = model.encode([r["text"] for r in rows], batch_size=256, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    summary = {"n_rows": len(rows), "by_source": defaultdict(int),
               "exact_text_groups": len(dup_groups(exact_text)),
               "exact_text_extra_rows": sum(len(g) - 1 for g in dup_groups(exact_text)),
               "same_scenario_name_groups": len(dup_groups(exact_name)),
               "nn_heldout_to_pool": {}}
    for r in rows:
        summary["by_source"][f'{r["behavior"]}/{r["source"]}/{r["split"]}'] += 1

    pairs = []
    thresholds = (0.70, 0.75, 0.80, 0.85, 0.90)
    by_beh = defaultdict(list)
    for i, r in enumerate(rows):
        by_beh[r["behavior"]].append(i)
    for beh, idx in sorted(by_beh.items()):
        idx = np.array(idx)
        hidx = idx[[rows[i]["split"] != "pool" for i in idx]]
        pidx = idx[[rows[i]["split"] == "pool" for i in idx]]
        sims = emb[hidx] @ emb[pidx].T
        stats = {}
        for src in sorted({rows[i]["source"] for i in pidx}):
            cols = np.array([rows[i]["source"] == src for i in pidx])
            mx = sims[:, cols].max(axis=1)
            stats[src] = {"n_pool": int(cols.sum()), "mean_max_cos": round(float(mx.mean()), 4),
                          **{f"share_ge_{t}": round(float((mx >= t).mean()), 4) for t in thresholds}}
        mx = sims.max(axis=1)
        stats["all"] = {"n_pool": int(len(pidx)), "mean_max_cos": round(float(mx.mean()), 4),
                        **{f"share_ge_{t}": round(float((mx >= t).mean()), 4) for t in thresholds}}
        summary["nn_heldout_to_pool"][beh] = stats
        top = np.argsort(-sims, axis=1)[:, : args.k]
        for a, hi in enumerate(hidx):
            for rank, b in enumerate(top[a]):
                pi = pidx[b]
                pairs.append({"behavior": beh, "heldout_id": rows[hi]["id"], "heldout_split": rows[hi]["split"],
                              "heldout_text": rows[hi]["text"], "pool_id": rows[pi]["id"],
                              "pool_source": rows[pi]["source"], "pool_text": rows[pi]["text"],
                              "rank": rank, "cos": round(float(sims[a, b]), 4)})

    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    with open(args.out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for x in pairs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_rows", "exact_text_groups", "exact_text_extra_rows",
                                              "same_scenario_name_groups")}))
    for beh, st in summary["nn_heldout_to_pool"].items():
        print(beh, {s: (v["n_pool"], v["mean_max_cos"], v["share_ge_0.8"]) for s, v in st.items()})
    print("pairs:", len(pairs))


if __name__ == "__main__":
    main()
