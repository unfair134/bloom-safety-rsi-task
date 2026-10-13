"""Frozen partitions with leakage-free held-out sets.

The pool is used as petri_bloom generated it; near-duplicates inside a partition are left alone.
Near-duplicate pairs (``edges.jsonl`` from ``curation.dedupe``) only constrain crossings:

1. Final test: per behavior, the first ``test_per_behavior`` candidates in SHA-256 order of
   (split seed, candidate id).
2. Feedback validation: per behavior, the next ``val_per_behavior`` in the same order that are not
   near any final-test item.
3. Seed: from the rest, excluding anything near a final-test or validation item, up to
   ceil(seed_total / n_behaviors) per behavior; any shortfall is filled one item at a time from
   behaviors with leftovers in sorted order.

Everything else is reserve, tagged with why it was not used. Near-duplicate links are checked
across behaviors too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PARTITIONS = ("final_test", "feedback_validation", "seed", "reserve")
HELD_OUT = ("final_test", "feedback_validation")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def _order_key(split_seed: int, candidate_id: str) -> str:
    return hashlib.sha256(f"{split_seed}:{candidate_id}".encode()).hexdigest()


def _neighbors(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for edge in edges:
        out.setdefault(edge["a"], set()).add(edge["b"])
        out.setdefault(edge["b"], set()).add(edge["a"])
    return out


def assign_partitions(
    roster: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    test_per_behavior: int = 100,
    val_per_behavior: int = 50,
    seed_total: int = 1000,
    split_seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    nbrs = _neighbors(edges)
    by_behavior: dict[str, list[dict[str, Any]]] = {}
    for row in roster:
        by_behavior.setdefault(row["behavior"], []).append(row)
    behaviors = sorted(by_behavior)
    ordered = {
        b: sorted(by_behavior[b], key=lambda r: _order_key(split_seed, r["candidate_id"])) for b in behaviors
    }
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in PARTITIONS}
    used: set[str] = set()
    near_test: set[str] = set()

    for b in behaviors:
        if len(ordered[b]) < test_per_behavior:
            raise ValueError(f"{b}: {len(ordered[b])} candidates (need {test_per_behavior} for final test)")
        picked = ordered[b][:test_per_behavior]
        out["final_test"].extend(picked)
        used.update(r["candidate_id"] for r in picked)
    for row in out["final_test"]:
        near_test |= nbrs.get(row["candidate_id"], set())

    short: list[str] = []
    for b in behaviors:
        eligible = [r for r in ordered[b] if r["candidate_id"] not in used and r["candidate_id"] not in near_test]
        if len(eligible) < val_per_behavior:
            short.append(f"{b}: {len(eligible)} eligible for validation (need {val_per_behavior})")
            continue
        picked = eligible[:val_per_behavior]
        out["feedback_validation"].extend(picked)
        used.update(r["candidate_id"] for r in picked)
    if short:
        raise ValueError("insufficient candidates for feedback validation:\n  " + "\n  ".join(short))
    near_held = set(near_test)
    for row in out["feedback_validation"]:
        near_held |= nbrs.get(row["candidate_id"], set())

    leftovers: dict[str, list[dict[str, Any]]] = {}
    for b in behaviors:
        leftovers[b] = []
        for r in ordered[b]:
            cid = r["candidate_id"]
            if cid in used:
                continue
            if cid in near_held:
                out["reserve"].append(r | {"reserve_reason": "near_held_out"})
            else:
                leftovers[b].append(r)
    available = sum(len(v) for v in leftovers.values())
    if available < seed_total:
        raise ValueError(f"only {available} candidates remain for seed (need {seed_total})")

    cap = math.ceil(seed_total / len(behaviors))
    taken = {b: min(cap, len(leftovers[b])) for b in behaviors}
    while sum(taken.values()) > seed_total:
        biggest = max(behaviors, key=lambda b: (taken[b], b))
        taken[biggest] -= 1
    while sum(taken.values()) < seed_total:
        for b in behaviors:
            if sum(taken.values()) == seed_total:
                break
            if taken[b] < len(leftovers[b]):
                taken[b] += 1
    for b in behaviors:
        out["seed"].extend(leftovers[b][: taken[b]])
        out["reserve"].extend(r | {"reserve_reason": "beyond_seed_quota"} for r in leftovers[b][taken[b] :])
    for name in PARTITIONS:
        out[name].sort(key=lambda r: (r["behavior"], r["candidate_id"]))
    return out


def verify_partitions(parts: dict[str, list[dict[str, Any]]], edges: list[dict[str, Any]]) -> None:
    home: dict[str, str] = {}
    for name in PARTITIONS:
        for row in parts[name]:
            cid = row["candidate_id"]
            if cid in home:
                raise ValueError(f"{cid} in both {home[cid]} and {name}")
            home[cid] = name
    for edge in edges:
        pa, pb = home.get(edge["a"]), home.get(edge["b"])
        if pa == pb or "reserve" in (pa, pb) or None in (pa, pb):
            continue
        if pa in HELD_OUT or pb in HELD_OUT:
            raise ValueError(f"near-duplicate {edge['a']} ({pa}) and {edge['b']} ({pb}) cross a held-out partition")


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["behavior"]] = counts.get(row["behavior"], 0) + 1
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roster", type=Path, required=True, help="roster.jsonl from curation.dedupe edges")
    parser.add_argument("--edges", type=Path, required=True, help="edges.jsonl from curation.dedupe edges")
    parser.add_argument("--inventory", type=Path, required=True, help="inventory.jsonl (adds seed_path and seed_text)")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--test-per-behavior", type=int, default=100)
    parser.add_argument("--val-per-behavior", type=int, default=50)
    parser.add_argument("--seed-total", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"{args.out} exists; partitions are frozen once written", file=sys.stderr)
        return 1
    inventory = {row["id"]: row for row in _read_jsonl(args.inventory)}
    roster = _read_jsonl(args.roster)
    edges = _read_jsonl(args.edges)
    try:
        parts = assign_partitions(
            roster,
            edges,
            test_per_behavior=args.test_per_behavior,
            val_per_behavior=args.val_per_behavior,
            seed_total=args.seed_total,
            split_seed=args.split_seed,
        )
        verify_partitions(parts, edges)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    manifest: dict[str, Any] = {
        "config": {
            "test_per_behavior": args.test_per_behavior,
            "val_per_behavior": args.val_per_behavior,
            "seed_total": args.seed_total,
            "split_seed": args.split_seed,
        },
        "roster_sha256": hashlib.sha256(args.roster.read_bytes()).hexdigest(),
        "edges_sha256": hashlib.sha256(args.edges.read_bytes()).hexdigest(),
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "partitions": {},
    }
    args.out.mkdir(parents=True)
    for name in PARTITIONS:
        rows = [
            row | {"seed_path": inventory[row["candidate_id"]]["seed_path"], "seed_text": inventory[row["candidate_id"]]["seed_text"]}
            for row in parts[name]
        ]
        text = _jsonl(rows)
        _atomic_write_text(args.out / f"{name}.jsonl", text)
        entry: dict[str, Any] = {
            "count": len(rows),
            "per_behavior": _counts(rows),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        if name == "reserve":
            reasons: dict[str, int] = {}
            for row in rows:
                reasons[row["reserve_reason"]] = reasons.get(row["reserve_reason"], 0) + 1
            entry["reasons"] = dict(sorted(reasons.items()))
        manifest["partitions"][name] = entry
    _atomic_write_text(args.out / "split_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["per_behavior"] for k, v in manifest["partitions"].items()}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
