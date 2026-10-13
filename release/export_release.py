"""Export a frozen dataset version as clean release files (no log metadata, paths or model routing).

    python export_release.py --dataset runs/dataset_v2_20261012 --behaviors runs/pool_v1_full_20261009 \
        --qwen-quality q1.jsonl q2.jsonl ... --claude-runs RUN_DIR=ROOT_DIR ... --out release_v2

public/seeds.jsonl     one row per seed: outline, quality scores and two reference trajectories
                       (claude_vs_qwen: Claude auditor vs. the base target; claude_vs_claude)
public/behaviors/      frozen BEHAVIOR.md, examples, understanding.md and dimensions per behavior
heldout/*.jsonl        final_test and feedback_validation outlines (to be encrypted before release)
manifest.json          counts and SHA-256 of every file

A trajectory keeps the final target conversation (Qwen reasoning attached to its turns; Claude's
reasoning is encrypted by the API and is not available), the auditor's turns with their realism
approval, the judge's scores and explanation. Only message content is exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

BEHAVIOR_FILES = ("BEHAVIOR.md", "scenarios/understanding.md")
BEHAVIOR_DIRS = ("examples", "scenarios/dimensions")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _text(m: Any) -> str:
    if isinstance(m.content, str):
        return m.content
    return "".join(c.text for c in m.content if getattr(c, "type", None) == "text")


def _reasoning(m: Any) -> str:
    if isinstance(m.content, str):
        return ""
    return "\n".join(c.reasoning for c in m.content
                     if getattr(c, "type", None) == "reasoning" and not c.redacted and c.reasoning)


def _msg(m: Any) -> dict:
    out = {"role": m.role, "content": _text(m)}
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [{"name": c.function, "arguments": dict(c.arguments)} for c in m.tool_calls]
    if m.role == "tool" and getattr(m, "error", None) is not None and not out["content"]:
        out["content"] = f"Error: {m.error.message}"
    return out


def trajectory(sample: Any) -> dict:
    events = list(sample.events)
    models = [e for e in events if type(e).__name__ == "ModelEvent" and not e.error and e.output and e.output.choices]
    rejected = {e.id for e in events if type(e).__name__ == "ToolEvent" and e.error is not None
                and e.error.type == "approval"}
    target = [e for e in models if e.role == "target"]
    conv: list[dict] = []
    if target:
        last = target[-1]
        conv = [_msg(m) for m in last.input] + [_msg(last.output.message)]
        reasoning_by_text = defaultdict(list)
        for e in target:
            r = _reasoning(e.output.message)
            if r:
                reasoning_by_text[_text(e.output.message)].append(r)
        for m in conv:
            if m["role"] == "assistant" and reasoning_by_text.get(m["content"]):
                m["reasoning"] = reasoning_by_text[m["content"]][-1]
    auditor = []
    for e in models:
        if e.role != "auditor":
            continue
        reply = e.output.message
        auditor.append({"content": _text(reply),
                        "tool_calls": [{"name": c.function, "arguments": dict(c.arguments),
                                        "realism_approved": c.id not in rejected} for c in reply.tool_calls or []]})
    score = (sample.scores or {}).get("audit_judge")
    value = dict(score.value) if score is not None and isinstance(score.value, dict) else None
    return {"target_conversation": conv, "auditor_turns": auditor, "judge_scores": value,
            "judge_explanation": score.explanation if score is not None else None,
            "judge_summary": (score.metadata or {}).get("summary") if score is not None else None}


def _read(job: tuple[str, list[tuple[str, str]]]) -> list[tuple[str, dict]]:
    from inspect_ai.log import read_eval_log
    path, wanted = job
    log = read_eval_log(path, resolve_attachments=True)
    by_id = {str(s.id): s for s in log.samples or []}
    return [(cid, trajectory(by_id[sid])) for cid, sid in wanted if sid in by_id]


def claude_jobs(specs: list[str]) -> dict[str, list[tuple[str, str]]]:
    """RUN_DIR=ROOT_DIR: logs of b1.evaluate runs and the materialized roots they ran on."""
    jobs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for spec in specs:
        run, root = map(Path, spec.split("=", 1))
        ids = {(r["behavior"], r["scenario_name"]): r["id"] for r in read_jsonl(root / "pilot_manifest.jsonl")}
        for log in sorted((run / "logs").glob("*.eval")):
            behavior = log.name.split("_bloom-", 1)[1].rsplit("_", 1)[0].replace("-", "_")
            for (b, name), cid in ids.items():
                if b == behavior:
                    jobs[str(log)].append((cid, name))
    return jobs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--behaviors", type=Path, required=True, help="pool root with the frozen behavior files")
    ap.add_argument("--qwen-quality", type=Path, nargs="+", required=True)
    ap.add_argument("--claude-runs", nargs="+", required=True,
                    help="RUN_DIR=ROOT_DIR; a later run overrides an earlier one for the same seed (retries)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    seeds = read_jsonl(args.dataset / "seed.jsonl")
    seed_ids = {r["candidate_id"] for r in seeds}
    qwen_jobs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    quality = {}
    for q in args.qwen_quality:
        for r in read_jsonl(q):
            if r["candidate_id"] in seed_ids and r["candidate_id"] not in quality:
                quality[r["candidate_id"]] = r
                qwen_jobs[r["log_path"]].append((r["candidate_id"], str(r["sample_id"])))
    cjobs = claude_jobs(args.claude_runs)
    cjobs = {k: [(c, s) for c, s in v if c in seed_ids] for k, v in cjobs.items()}

    def run(jobs: dict) -> dict[str, dict]:
        out = {}
        with ProcessPoolExecutor(args.workers) as pool:
            for rows in pool.map(_read, [(k, v) for k, v in jobs.items() if v]):
                out.update(rows)
        return out

    vs_qwen, vs_claude = run(qwen_jobs), run(cjobs)
    missing = {"claude_vs_qwen": sorted(seed_ids - set(vs_qwen)), "claude_vs_claude": sorted(seed_ids - set(vs_claude))}
    if any(missing.values()):
        raise SystemExit(f"missing trajectories: { {k: len(v) for k, v in missing.items()} } e.g. {missing}")
    empty = [cid for d in (vs_qwen, vs_claude) for cid, t in d.items()
             if not any(m["role"] == "assistant" for m in t["target_conversation"])]
    if empty:
        raise SystemExit(f"trajectories without a target reply (rerun them): {empty}")

    pub, held = args.out / "public", args.out / "heldout"
    pub.mkdir(parents=True)
    held.mkdir()
    with open(pub / "seeds.jsonl", "w", encoding="utf-8") as f:
        for r in seeds:
            cid = r["candidate_id"]
            f.write(json.dumps({
                "id": cid, "behavior": r["behavior"], "scenario_name": Path(r["seed_path"]).stem,
                "outline": r["seed_text"],
                "quality_check": {"scenario_realism": r["scenario_realism"], "eval_awareness": r["eval_awareness"]},
                "claude_vs_qwen": vs_qwen[cid], "claude_vs_claude": vs_claude[cid],
            }, ensure_ascii=False) + "\n")
    for b in sorted({r["behavior"] for r in seeds}):
        for rel in BEHAVIOR_FILES:
            dst = pub / "behaviors" / b / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.behaviors / b / rel, dst)
        for rel in BEHAVIOR_DIRS:
            shutil.copytree(args.behaviors / b / rel, pub / "behaviors" / b / rel)
    for name in ("final_test", "feedback_validation"):
        with open(held / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in read_jsonl(args.dataset / f"{name}.jsonl"):
                f.write(json.dumps({"id": r["candidate_id"], "behavior": r["behavior"],
                                    "scenario_name": Path(r["seed_path"]).stem, "outline": r["seed_text"],
                                    "quality_check": {"scenario_realism": r["scenario_realism"],
                                                      "eval_awareness": r["eval_awareness"]}},
                                   ensure_ascii=False) + "\n")
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    manifest = {"dataset_version": json.loads((args.dataset / "VERSION.json").read_text())["version"],
                "files": {str(p.relative_to(args.out)): sha256(p) for p in files}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"seeds": len(seeds), "files": len(files)}))


if __name__ == "__main__":
    main()
