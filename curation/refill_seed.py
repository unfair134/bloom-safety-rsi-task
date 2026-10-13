"""Seed v2: drop v1 seeds that duplicate a held-out item, refill to an equal count per behavior.

A behavior left above the quota drops its last v1 seeds in SHA-256(split_seed:id) order.

Held-out partitions (final_test, feedback_validation) are copied from v1 unchanged.

pairs   Candidates = accepted, unused, not leak-flagged pool rows, in SHA-256(split_seed:id) order;
        the first `--spare` x shortfall per behavior are screened. For each candidate, its top-k
        nearest held-out items, kept seeds and earlier candidates (same behavior, mpnet cosine)
        become pairs for label_pairs.py. Cosine only retrieves pairs; Claude decides.
build   Walk candidates in order; accept one when none of its labeled pairs is a duplicate of a
        held-out item, a kept seed or an already accepted candidate. Writes the v2 dataset dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HELD_OUT = ("final_test", "feedback_validation")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def order_key(seed: int, cid: str) -> str:
    return hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()


def leak_flags(dup_dir: Path) -> set[str]:
    """Non-held-out ids judged the same scenario as some held-out item."""
    labels = {}
    for r in read_jsonl(dup_dir / "labels.jsonl"):
        labels[r["key"]] = r["duplicate"]
    flagged = set()
    for p in read_jsonl(dup_dir / "pairs.jsonl"):
        k = f'{p["heldout_id"]}:{p["pool_id"]}'
        if k not in labels or labels[k] is None:
            raise SystemExit(f"unlabeled pair {k} in {dup_dir}")
        if labels[k]:
            flagged.add(p["pool_id"])
    return flagged


def load_candidates(pools: list[str], used: set[str], flagged: set[str]) -> list[dict]:
    """pools: name=quality.jsonl=inventory.jsonl"""
    out, seen = [], set()
    for spec in pools:
        name, qpath, ipath = spec.split("=", 2)
        inv = {r["id"]: r for r in read_jsonl(Path(ipath))}
        for r in read_jsonl(Path(qpath)):
            cid = r["candidate_id"]
            if not r["accepted"] or cid in used or cid in flagged or cid in seen:
                continue
            seen.add(cid)
            s = r["official_scores"]
            out.append({"candidate_id": cid, "behavior": r["behavior"],
                        "scenario_realism": s["scenario_realism"], "eval_awareness": s["eval_awareness"],
                        "source_pool": name, "seed_path": inv[cid]["seed_path"], "seed_text": inv[cid]["seed_text"]})
    return out


def common(args) -> dict:
    v1 = {name: read_jsonl(args.v1 / f"{name}.jsonl") for name in (*HELD_OUT, "seed", "reserve")}
    flagged = leak_flags(args.dup)
    held = {r["candidate_id"]: r for n in HELD_OUT for r in v1[n]}
    kept = [r | {"source_pool": "v1"} for r in v1["seed"] if r["candidate_id"] not in flagged]
    dropped = [r | {"drop_reason": "duplicates_held_out"} for r in v1["seed"] if r["candidate_id"] in flagged]
    for b in sorted({r["behavior"] for r in kept}):
        mine = sorted((r for r in kept if r["behavior"] == b), key=lambda r: order_key(args.split_seed, r["candidate_id"]))
        for r in mine[args.per_behavior:]:
            kept.remove(r)
            dropped.append(r | {"drop_reason": "over_quota"})
    used = set(held) | {r["candidate_id"] for r in v1["seed"]}
    cands = load_candidates(args.pool, used, flagged)
    need = {}
    for b in sorted({r["behavior"] for r in held.values()}):
        need[b] = args.per_behavior - sum(r["behavior"] == b for r in kept)
    by_b = defaultdict(list)
    for c in sorted(cands, key=lambda r: order_key(args.split_seed, r["candidate_id"])):
        by_b[c["behavior"]].append(c)
    screened = {b: by_b[b][: max(0, need[b]) * args.spare] for b in need}
    return {"v1": v1, "held": held, "kept": kept, "dropped": dropped, "need": need,
            "screened": screened, "available": {b: len(by_b[b]) for b in need}, "flagged": flagged}


def cmd_pairs(args) -> None:
    st = common(args)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(args.model), device="cuda")
    enc = lambda xs: model.encode(xs, batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    pairs = []
    for b, cands in st["screened"].items():
        if not cands:
            continue
        refs = [("heldout", r) for r in st["held"].values() if r["behavior"] == b]
        refs += [("seed", r) for r in st["kept"] if r["behavior"] == b]
        ce = enc([c["seed_text"] for c in cands])
        re_ = enc([r["seed_text"] for _, r in refs])
        sims = ce @ re_.T
        cc = ce @ ce.T
        for i, c in enumerate(cands):
            for j in np.argsort(-sims[i])[: args.k]:
                kind, r = refs[j]
                pairs.append(dict(behavior=b, heldout_id=r["candidate_id"], heldout_split=kind,
                                  heldout_text=r["seed_text"], pool_id=c["candidate_id"],
                                  pool_source=c["source_pool"], pool_text=c["seed_text"], cos=round(float(sims[i, j]), 4)))
            if i:
                for j in np.argsort(-cc[i, :i])[: args.k]:
                    r = cands[j]
                    pairs.append(dict(behavior=b, heldout_id=r["candidate_id"], heldout_split="candidate",
                                      heldout_text=r["seed_text"], pool_id=c["candidate_id"],
                                      pool_source=c["source_pool"], pool_text=c["seed_text"], cos=round(float(cc[i, j]), 4)))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"dropped_v1_seeds": len(st["dropped"]), "need": st["need"], "available": st["available"],
                      "screened": {b: len(v) for b, v in st["screened"].items()}, "pairs": len(pairs)}))


def cmd_build(args) -> None:
    st = common(args)
    labels = {r["key"]: r["duplicate"] for r in read_jsonl(args.out / "labels.jsonl")}
    by_cand = defaultdict(list)
    for p in read_jsonl(args.out / "pairs.jsonl"):
        k = f'{p["heldout_id"]}:{p["pool_id"]}'
        if labels.get(k) is None:
            raise SystemExit(f"pair {k} unlabeled or errored; rerun label_pairs.py")
        by_cand[p["pool_id"]].append((p["heldout_split"], p["heldout_id"], labels[k]))
    added, rejected = [], []
    for b, cands in st["screened"].items():
        accepted_ids: set[str] = set()
        for c in cands:
            if len(accepted_ids) == st["need"][b]:
                break
            hits = [(kind, oid) for kind, oid, dup in by_cand[c["candidate_id"]]
                    if dup and (kind != "candidate" or oid in accepted_ids)]
            if hits:
                rejected.append(c | {"duplicate_of": hits})
            else:
                accepted_ids.add(c["candidate_id"])
                added.append(c)
        if len(accepted_ids) < st["need"][b]:
            raise SystemExit(f"{b}: only {len(accepted_ids)}/{st['need'][b]} accepted; raise --spare")
    seed = sorted(st["kept"] + added, key=lambda r: (r["behavior"], r["candidate_id"]))
    dst = args.dataset_out
    if dst.exists():
        raise SystemExit(f"{dst} exists; datasets are frozen once written")
    dst.mkdir(parents=True)
    manifest = {"config": {"per_behavior": args.per_behavior, "split_seed": args.split_seed, "k": args.k, "spare": args.spare},
                "base_version": str(args.v1), "partitions": {}}

    def write(name: str, rows: list[dict]) -> None:
        text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
        path = dst / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        per = defaultdict(int)
        for r in rows:
            per[r["behavior"]] += 1
        manifest["partitions"][name] = {"count": len(rows), "per_behavior": dict(sorted(per.items())),
                                        "sha256": hashlib.sha256(text.encode()).hexdigest()}

    for name in HELD_OUT:
        write(name, st["v1"][name])
    write("seed", seed)
    write("seed_v1_dropped", sorted(st["dropped"], key=lambda r: (r["behavior"], r["candidate_id"])))
    write("refill_rejected", rejected)
    (dst / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["per_behavior"] for k, v in manifest["partitions"].items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pairs", "build"))
    ap.add_argument("--v1", type=Path, required=True, help="frozen v1 dataset dir")
    ap.add_argument("--dup", type=Path, required=True, help="dup_scale dir with fully labeled pairs/labels")
    ap.add_argument("--pool", nargs="+", required=True, help="name=quality.jsonl=inventory.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="work dir for pairs.jsonl / labels.jsonl")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--dataset-out", type=Path)
    ap.add_argument("--per-behavior", type=int, default=167)
    ap.add_argument("--split-seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--spare", type=int, default=3)
    args = ap.parse_args()
    cmd_pairs(args) if args.cmd == "pairs" else cmd_build(args)


if __name__ == "__main__":
    main()
