"""Create deterministic, blinded HTML audits from ``quality.jsonl``.

Case pages intentionally omit behavior scores and explicit acceptance labels.
The keyed answer manifest is separate from the pages and public case manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "bloom-quality-audit-v1"


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_quality(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id or candidate_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate or missing candidate_id")
            seen.add(candidate_id)
            rows.append(row)
    return rows


def _selection_hash(row: Mapping[str, Any]) -> str:
    payload = {
        "schema": AUDIT_SCHEMA,
        "behavior": str(row["behavior"]),
        "candidate_id": str(row["candidate_id"]),
        "outcome": bool(row["accepted"]),
    }
    return _canonical_hash(payload)


def select_audit_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_outcome: int = 20,
) -> list[dict[str, Any]]:
    if per_outcome < 1:
        raise ValueError("per_outcome must be at least 1")
    groups: dict[tuple[str, bool], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"duplicate or missing candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        groups[(str(row["behavior"]), bool(row["accepted"]))].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], not item[1])):
        ranked = sorted(
            groups[key],
            key=lambda row: (_selection_hash(row), str(row["candidate_id"])),
        )
        for row in ranked[:per_outcome]:
            digest = _selection_hash(row)
            selected.append(
                {
                    "case_id": f"case-{digest[:20]}",
                    "selection_sha256": digest,
                    "row": row,
                }
            )
    return sorted(selected, key=lambda item: item["case_id"])


def _render_transcript(transcript: Any) -> str:
    blocks: list[str] = []
    if not isinstance(transcript, list):
        transcript = []
    for message in transcript:
        if not isinstance(message, Mapping):
            continue
        role = html.escape(str(message.get("role", "unknown")))
        content = html.escape(str(message.get("content", "")))
        blocks.append(
            '<section class="message">'
            f"<h3>{role}</h3>"
            f"<pre>{content}</pre>"
            "</section>"
        )
    return "\n".join(blocks) or "<p>No transcript available.</p>"


def render_case_html(case_id: str, row: Mapping[str, Any]) -> str:
    scores = row.get("official_scores", {})
    if not isinstance(scores, Mapping):
        scores = {}
    reasons = row.get("rejection_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    reason_html = (
        "<ul>"
        + "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reasons)
        + "</ul>"
        if reasons
        else "<p>None recorded.</p>"
    )
    scenario = html.escape(str(row.get("scenario", "")))
    realism = html.escape(str(scores.get("scenario_realism")))
    awareness = html.escape(str(scores.get("eval_awareness")))
    behavior = html.escape(str(row.get("behavior", "")))
    escaped_case_id = html.escape(case_id)
    transcript = _render_transcript(row.get("transcript"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Quality audit {escaped_case_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem auto; max-width: 70rem; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5; padding: 1rem; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #bbb; padding: .4rem .7rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>Quality audit case {escaped_case_id}</h1>
  <p><strong>Behavior:</strong> {behavior}</p>
  <h2>Scenario</h2>
  <pre>{scenario}</pre>
  <h2>Transcript</h2>
  {transcript}
  <h2>Official quality scores</h2>
  <table>
    <tr><th>Scenario realism</th><td>{realism}</td></tr>
    <tr><th>Evaluation awareness</th><td>{awareness}</td></tr>
  </table>
  <h2>Rejection reasons</h2>
  {reason_html}
</body>
</html>
"""


def build_audit_bundle(
    *,
    quality_path: Path,
    out_dir: Path,
    per_outcome: int = 20,
) -> dict[str, Any]:
    quality_path = Path(quality_path).resolve()
    out_dir = Path(out_dir)
    rows = _read_quality(quality_path)
    selected = select_audit_cases(rows, per_outcome=per_outcome)

    config_core = {
        "schema": AUDIT_SCHEMA,
        "quality_path": str(quality_path),
        "quality_sha256": _sha256_file(quality_path),
        "per_behavior_accepted": per_outcome,
        "per_behavior_rejected": per_outcome,
        "selection": "lowest_sha256_by_behavior_and_outcome",
    }
    config = {**config_core, "config_sha256": _canonical_hash(config_core)}
    config_path = out_dir / "audit.config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"{out_dir}: audit config differs from existing bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(config_path, config)

    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    expected_pages: set[str] = set()
    public_cases: list[dict[str, Any]] = []
    answer_cases: list[dict[str, Any]] = []
    for item in selected:
        case_id = item["case_id"]
        row = item["row"]
        page_name = f"{case_id}.html"
        expected_pages.add(page_name)
        page_path = cases_dir / page_name
        _atomic_write_text(page_path, render_case_html(case_id, row))
        public_cases.append(
            {
                "case_id": case_id,
                "behavior": str(row["behavior"]),
                "page": f"cases/{page_name}",
                "page_sha256": _sha256_file(page_path),
            }
        )
        answer_cases.append(
            {
                "case_id": case_id,
                "candidate_id": str(row["candidate_id"]),
                "behavior": str(row["behavior"]),
                "scenario_name": str(row.get("scenario_name", "")),
                "sample_id": str(row.get("sample_id", "")),
                "log_path": str(row.get("log_path", "")),
                "accepted": bool(row["accepted"]),
                "rejection_reasons": list(row.get("rejection_reasons", [])),
                "selection_sha256": item["selection_sha256"],
            }
        )
    for stale in cases_dir.glob("*.html"):
        if stale.name not in expected_pages:
            stale.unlink()

    answers = {
        "schema": AUDIT_SCHEMA,
        "config_sha256": config["config_sha256"],
        "cases": answer_cases,
    }
    _atomic_write_json(out_dir / "answer_key.json", answers)
    manifest = {
        "schema": AUDIT_SCHEMA,
        "config_sha256": config["config_sha256"],
        "cases": public_cases,
        "answer_key_sha256": _sha256_file(out_dir / "answer_key.json"),
    }
    _atomic_write_json(out_dir / "audit.manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select deterministic blind quality-audit cases and render escaped HTML."
    )
    parser.add_argument("--quality", type=Path, required=True, help="quality.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="audit output directory")
    parser.add_argument(
        "--per-outcome",
        type=int,
        default=20,
        help="maximum accepted and rejected cases per behavior (default: 20 each)",
    )
    args = parser.parse_args(argv)
    build_audit_bundle(
        quality_path=args.quality,
        out_dir=args.out,
        per_outcome=args.per_outcome,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
