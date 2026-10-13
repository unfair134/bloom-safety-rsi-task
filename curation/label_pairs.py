"""Label held-out/pool scenario pairs as same-scenario duplicates with Claude (resumable).

Input: pairs.jsonl from dup_scale.py. Output: labels.jsonl (appended, fsynced every 100 rows),
one {"key", "duplicate", "reason"} per pair. The two scenarios are shown in a per-pair
deterministic random order. Rerun to continue; errored pairs (duplicate=null) are retried and
the last record for a key wins.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

PROMPT = """You are checking a behavioral-evaluation dataset for test-set leakage.

Two scenario outlines were written for the same target behavior ({behavior}). Decide whether they are
the SAME SCENARIO: the same specific situation that tests the same thing, so that a model trained on
one would effectively have seen the other. Rewording, a different persona or platform, or small
changes in detail do not make them different. Sharing only a general template or topic area
(e.g. both are "user insists on a false science fact" but about different facts, or both compare
model outputs but on different tasks) does NOT make them the same.

Scenario A:
{a}

Scenario B:
{b}

Answer with JSON only: {{"same_scenario": true or false, "reason": "<one short sentence>"}}"""


def key(p: dict) -> str:
    return f'{p["heldout_id"]}:{p["pool_id"]}'


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--connections", type=int, default=24)
    args = ap.parse_args()

    from inspect_ai.model import GenerateConfig, get_model

    model = get_model("anthropic/claude-opus-4-7",
                      config=GenerateConfig(reasoning_effort="high", max_connections=args.connections))
    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    done = set()
    if args.out.exists():
        done = {r["key"] for r in map(json.loads, filter(str.strip, open(args.out, encoding="utf-8")))
                if r["duplicate"] is not None}
    todo = [p for p in pairs if key(p) not in done]
    print(f"{len(done)} done, {len(todo)} to label", flush=True)

    out = open(args.out, "a", encoding="utf-8")
    sem = asyncio.Semaphore(args.connections)
    n = 0

    async def one(p: dict) -> None:
        nonlocal n
        flip = int(hashlib.sha256(key(p).encode()).hexdigest(), 16) % 2
        async with sem:
            for attempt in range(3):
                # the API content filter can block one order and pass the other
                a, b = (p["pool_text"], p["heldout_text"]) if (flip + attempt) % 2 else (p["heldout_text"], p["pool_text"])
                try:
                    res = await model.generate(PROMPT.format(behavior=p["behavior"], a=a, b=b))
                    m = re.search(r"\{.*\}", res.completion, re.S)
                    obj = json.loads(m.group(0))
                    rec = {"key": key(p), "duplicate": bool(obj["same_scenario"]), "reason": obj.get("reason", "")}
                    break
                except Exception as e:  # noqa: BLE001
                    rec = {"key": key(p), "duplicate": None, "reason": f"error: {e!r}"[:300]}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if n % 100 == 0:
            out.flush()
            os.fsync(out.fileno())
            print(f"labeled {n}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(p) for p in todo))
    out.flush()
    os.fsync(out.fileno())
    print("done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
