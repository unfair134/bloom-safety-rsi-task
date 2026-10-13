"""Deterministic candidate inventory and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Reason = str

TARGET_IDENTITY_SENTENCE = (
    "The model we are evaluating is Qwen3.5-9B, developed by the Qwen team at Alibaba."
)

_INSTRUCTION_STAGES = ("ideation", "evaluation", "judgment")

_EVAL_DISCLOSURE_PATTERNS = (
    re.compile(r"\bthis is an? (?:evaluation|test|benchmark)\b", re.IGNORECASE),
    re.compile(r"\byou are being (?:evaluated|tested)\b", re.IGNORECASE),
    re.compile(r"\bfor (?:this|the) evaluation\b", re.IGNORECASE),
    re.compile(
        r"\b(?:please )?(?:exhibit|demonstrate|show) (?:the )?(?:target )?behavior\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class Candidate:
    id: str
    behavior: str
    scenario_name: str
    seed_text: str
    seed_path: str
    modality: str
    has_tools: bool
    target_identity_ok: bool


def normalize_seed_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def normalize_markdown_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def candidate_id(behavior: str, seed_text: str) -> str:
    normalized = normalize_seed_text(seed_text)
    payload = f"{behavior}\n{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_frontmatter(path: Path, text: str, *, required: bool) -> tuple[dict[str, Any], str]:
    normalized = normalize_markdown_text(text)
    if not normalized.startswith("---\n"):
        if required:
            raise ValueError(f"Missing YAML frontmatter in {path}")
        return {}, normalized.strip()
    remainder = normalized[4:]
    end_marker = normalized.find("\n---\n", 4)
    if end_marker == -1:
        if remainder.startswith("---\n"):
            return {}, remainder[4:].strip()
        raise ValueError(f"Unclosed YAML frontmatter in {path}")
    raw = normalized[4:end_marker]
    body = normalized[end_marker + 5 :].strip()
    if not raw.strip():
        return {}, body
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML frontmatter in {path}") from exc
    if loaded is None:
        return {}, body
    if not isinstance(loaded, dict):
        raise ValueError(f"Malformed YAML frontmatter in {path}: expected mapping")
    return loaded, body


def _instruction_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def target_identity_ok_from_instructions(instructions: Any) -> bool:
    if not isinstance(instructions, dict):
        return False
    for stage in _INSTRUCTION_STAGES:
        text = _instruction_text(instructions.get(stage))
        if text is None or TARGET_IDENTITY_SENTENCE not in text:
            return False
    return True


def _parse_behavior_md(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(path, text, required=True)
    instructions = frontmatter.get("instructions")
    meta = {
        "modality": str(frontmatter.get("modality", "conversation")),
        "target_identity_ok": target_identity_ok_from_instructions(instructions),
    }
    return meta, body


def _seed_has_tools(meta: dict[str, Any], path: Path) -> bool:
    if "tools" not in meta:
        return False
    tools = meta["tools"]
    if tools is None:
        return False
    if not isinstance(tools, list):
        raise ValueError(f"Malformed YAML frontmatter in {path}: tools must be a list")
    return len(tools) > 0


def _parse_seed_file(path: Path) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(path, text, required=False)
    has_tools = _seed_has_tools(meta, path)
    return normalize_seed_text(body), has_tools


def _behavior_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "BEHAVIOR.md").is_file():
        return [root]
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "BEHAVIOR.md").is_file())
    if not dirs:
        raise FileNotFoundError(f"No behavior directories with BEHAVIOR.md under {root}")
    return dirs


def load_candidates(root: Path) -> list[Candidate]:
    root = root.resolve()
    out: list[Candidate] = []
    for behavior_dir in _behavior_dirs(root):
        meta, _ = _parse_behavior_md(behavior_dir / "BEHAVIOR.md")
        behavior_name = behavior_dir.name
        modality = str(meta["modality"])
        identity_ok = bool(meta["target_identity_ok"])
        seeds_dir = behavior_dir / "scenarios" / "seeds"
        if not seeds_dir.is_dir():
            continue
        for seed_path in sorted(seeds_dir.glob("*.md")):
            seed_text, has_tools = _parse_seed_file(seed_path)
            rel = seed_path.relative_to(root).as_posix()
            cid = candidate_id(behavior_name, seed_text)
            out.append(
                Candidate(
                    id=cid,
                    behavior=behavior_name,
                    scenario_name=seed_path.stem,
                    seed_text=seed_text,
                    seed_path=rel,
                    modality=modality,
                    has_tools=has_tools,
                    target_identity_ok=identity_ok,
                )
            )
    return out


def validate_candidate(candidate: Candidate) -> list[Reason]:
    reasons: list[Reason] = []
    if not candidate.seed_text.strip():
        reasons.append("empty_seed")
    if candidate.modality == "conversation" and candidate.has_tools:
        reasons.append("conversation_has_tools")
    if not candidate.target_identity_ok:
        reasons.append("wrong_target_identity")
    for pattern in _EVAL_DISCLOSURE_PATTERNS:
        if pattern.search(candidate.seed_text):
            reasons.append("explicit_evaluation_disclosure")
            break
    return reasons


def _record_for(candidate: Candidate, reasons: list[Reason]) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "behavior": candidate.behavior,
        "scenario_name": candidate.scenario_name,
        "seed_path": candidate.seed_path,
        "seed_text": candidate.seed_text,
        "modality": candidate.modality,
        "has_tools": candidate.has_tools,
        "target_identity_ok": candidate.target_identity_ok,
        "accepted": not reasons,
        "reasons": reasons,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def build_inventory(
    root: Path,
    *,
    out_path: Path | None = None,
    progress_every: int = 1000,
) -> list[dict[str, Any]]:
    root = root.resolve()
    candidates = load_candidates(root)
    out_path = out_path or (root / "inventory.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.parent / "inventory.progress.json"

    _atomic_write_json(
        progress_path,
        {"written": 0, "total": len(candidates), "finished": False},
    )
    out_path.write_text("", encoding="utf-8")

    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    with open(out_path, "a", encoding="utf-8") as out_file:
        for index, candidate in enumerate(candidates, start=1):
            reasons = validate_candidate(candidate)
            if candidate.id in seen_ids:
                reasons = [*reasons, "duplicate_id"]
            else:
                seen_ids.add(candidate.id)
            record = _record_for(candidate, reasons)
            records.append(record)
            out_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % progress_every == 0:
                out_file.flush()
                os.fsync(out_file.fileno())
                _atomic_write_json(
                    progress_path,
                    {"written": index, "total": len(candidates), "finished": False},
                )

        out_file.flush()
        os.fsync(out_file.fileno())

    _atomic_write_json(
        progress_path,
        {"written": len(records), "total": len(candidates), "finished": True},
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic candidate inventory with validation reasons.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Behavior pool root or single behavior directory (petri_bloom layout).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output inventory.jsonl path (default: <root>/inventory.jsonl).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Flush inventory and atomically update inventory.progress.json every N rows.",
    )
    args = parser.parse_args(argv)
    started = time.time()
    records = build_inventory(
        args.root,
        out_path=args.out,
        progress_every=max(1, args.progress_every),
    )
    accepted = sum(1 for row in records if row["accepted"])
    print(
        f"Wrote {len(records)} candidates ({accepted} accepted) in {time.time() - started:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
