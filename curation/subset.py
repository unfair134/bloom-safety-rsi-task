"""Deterministic pilot subset selection and petri_bloom directory materialization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from curation.inventory import _parse_seed_file, candidate_id

_MANIFEST_FIELDS = (
    "id",
    "behavior",
    "scenario_name",
    "seed_path",
    "seed_text",
    "modality",
    "has_tools",
    "target_identity_ok",
)


def load_inventory(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_pilot_candidates(
    records: list[dict[str, Any]],
    *,
    per_behavior: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Select the first ``per_behavior`` accepted rows per behavior in SHA-256 order; ``None`` selects all."""
    if per_behavior is not None and per_behavior < 1:
        raise ValueError("per_behavior must be at least 1")
    by_behavior: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if not row.get("accepted"):
            continue
        by_behavior.setdefault(str(row["behavior"]), []).append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    for behavior in sorted(by_behavior):
        rows = sorted(
            by_behavior[behavior],
            key=lambda item: (str(item["id"]), str(item.get("scenario_name", ""))),
        )
        selected[behavior] = rows if per_behavior is None else rows[:per_behavior]
    return selected


def _is_under_or_same(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_no_source_output_overlap(source_root: Path, out_root: Path) -> None:
    if _is_under_or_same(source_root, out_root) or _is_under_or_same(out_root, source_root):
        raise ValueError(f"source and output paths overlap: {source_root} vs {out_root}")


def _assert_fresh_output(out_root: Path) -> None:
    if out_root.exists():
        raise ValueError(f"output root already exists: {out_root}; choose a fresh path")


def _assert_exact_per_behavior(records: list[dict[str, Any]], per_behavior: int) -> None:
    behaviors: set[str] = set()
    accepted_counts: dict[str, int] = {}
    for row in records:
        name = str(row["behavior"])
        behaviors.add(name)
        if row.get("accepted"):
            accepted_counts[name] = accepted_counts.get(name, 0) + 1
    shortfalls = [
        f"{name}: {accepted_counts.get(name, 0)} accepted (need {per_behavior})"
        for name in sorted(behaviors)
        if accepted_counts.get(name, 0) < per_behavior
    ]
    if shortfalls:
        raise ValueError(
            "insufficient accepted candidates for --per-behavior "
            f"{per_behavior}:\n  " + "\n  ".join(shortfalls)
        )


def _source_seed_id(behavior: str, seed_path: Path) -> str:
    seed_text, _has_tools = _parse_seed_file(seed_path)
    return candidate_id(behavior, seed_text)


def _required_artifact_issues(
    source_root: Path,
    behavior: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    src_dir = source_root / behavior
    issues: list[str] = []
    if not (src_dir / "BEHAVIOR.md").is_file():
        issues.append("BEHAVIOR.md")
    if not (src_dir / "examples").is_dir():
        issues.append("examples/")
    if not (src_dir / "scenarios" / "understanding.md").is_file():
        issues.append("scenarios/understanding.md")
    if not (src_dir / "scenarios" / "dimensions").exists():
        issues.append("scenarios/dimensions")
    for row in rows:
        rel = Path(str(row["seed_path"]))
        if not (source_root / rel).is_file():
            issues.append(rel.as_posix())
    return issues


def _assert_preflight(
    source_root: Path,
    selected: dict[str, list[dict[str, Any]]],
) -> None:
    missing: list[str] = []
    mismatches: list[str] = []
    for behavior in sorted(selected):
        rows = selected[behavior]
        for issue in _required_artifact_issues(source_root, behavior, rows):
            missing.append(f"{behavior}: {issue}")
        for row in rows:
            rel = Path(str(row["seed_path"]))
            seed_src = source_root / rel
            if not seed_src.is_file():
                continue
            actual = _source_seed_id(behavior, seed_src)
            if actual != row["id"]:
                mismatches.append(
                    f"{rel.as_posix()} inventory={row['id']} source={actual}"
                )
    if missing:
        raise ValueError("missing required artifacts:\n  " + "\n  ".join(missing))
    if mismatches:
        raise ValueError(
            "inventory/source candidate id mismatch:\n  " + "\n  ".join(mismatches)
        )


def _copy_bytes(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _copy_tree_bytes(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_file():
        _copy_bytes(src, dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _copy_bytes(path, target)


def _materialize_behavior(
    source_root: Path,
    out_root: Path,
    behavior: str,
    rows: list[dict[str, Any]],
) -> None:
    src_dir = source_root / behavior
    dst_dir = out_root / behavior

    _copy_bytes(src_dir / "BEHAVIOR.md", dst_dir / "BEHAVIOR.md")
    _copy_tree_bytes(src_dir / "examples", dst_dir / "examples")
    _copy_bytes(
        src_dir / "scenarios" / "understanding.md",
        dst_dir / "scenarios" / "understanding.md",
    )
    _copy_tree_bytes(src_dir / "scenarios" / "dimensions", dst_dir / "scenarios" / "dimensions")

    seeds_dst = dst_dir / "scenarios" / "seeds"
    seeds_dst.mkdir(parents=True, exist_ok=True)
    for row in rows:
        rel = Path(str(row["seed_path"]))
        seed_src = source_root / rel
        actual = _source_seed_id(behavior, seed_src)
        if actual != row["id"]:
            raise ValueError(
                "inventory/source candidate id mismatch: "
                f"{rel.as_posix()} inventory={row['id']} source={actual}"
            )
        _copy_bytes(seed_src, seeds_dst / rel.name)


def _manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in _MANIFEST_FIELDS}


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def build_pilot_subset(
    *,
    inventory_path: Path,
    source_root: Path,
    out_root: Path,
    per_behavior: int | None,
) -> list[dict[str, Any]]:
    if per_behavior is not None and per_behavior < 1:
        raise ValueError("per_behavior must be at least 1")
    source_root = source_root.resolve()
    out_root = out_root.resolve()
    inventory_path = inventory_path.resolve()

    _assert_no_source_output_overlap(source_root, out_root)
    _assert_fresh_output(out_root)

    records = load_inventory(inventory_path)
    if per_behavior is not None:
        _assert_exact_per_behavior(records, per_behavior)
    selected = select_pilot_candidates(records, per_behavior=per_behavior)
    _assert_preflight(source_root, selected)

    out_root.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, Any]] = []
    for behavior in sorted(selected):
        rows = selected[behavior]
        _materialize_behavior(source_root, out_root, behavior, rows)
        manifest.extend(_manifest_row(row) for row in rows)

    manifest_path = out_root / "pilot_manifest.jsonl"
    lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in manifest
    ]
    _atomic_write_text(manifest_path, "".join(lines))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible pilot subset from an accepted inventory.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="Path to inventory.jsonl from curation.inventory.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="petri_bloom behavior pool root used to build the inventory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for pilot behaviors and pilot_manifest.jsonl.",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--per-behavior",
        type=int,
        default=20,
        help="Exactly this many accepted candidates required per inventory behavior (default: 20).",
    )
    size.add_argument(
        "--all",
        action="store_true",
        help="Take every accepted candidate (counts may differ across behaviors).",
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_pilot_subset(
            inventory_path=args.inventory,
            source_root=args.source,
            out_root=args.out,
            per_behavior=None if args.all else args.per_behavior,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    behaviors = len({row["behavior"] for row in manifest})
    print(
        f"Wrote {len(manifest)} pilot seeds across {behaviors} behaviors to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
