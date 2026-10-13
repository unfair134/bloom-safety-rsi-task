"""One evaluation pass on a behavior root: auditor and target are Claude or a Qwen endpoint; Claude judges.

    AUDITOR_BASE_URL=... TARGET_BASE_URL=... ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... \\
    python -m judge.evaluate run --behaviors <root> --out <new_dir> --auditor qwen --target qwen \\
        [--auditor-model attacker] [--target-model qwen3.5-9b] [--resume]

Everything except the auditor and target roles matches the official quality rollout
(`curation.quality_rollout`): petri_bloom `bloom_audit`, 5 turns, realism filter 0.6, Claude judge and
realism approver, Qwen model-card sampling. A Qwen role reads its endpoint from AUDITOR_BASE_URL or
TARGET_BASE_URL and its served model name from --auditor-model / --target-model (default `qwen3.5-9b`,
the base; a LoRA adapter is served under its own name with vLLM `--lora-modules`). A Claude role uses
the quality-rollout Claude config. The behavior root must name the target (`identity` rewrites the
Qwen identity line for a Claude target).

Defender side: --auditor claude --target qwen --target-model <defender adapter>.
Attacker side: --auditor qwen --auditor-model <attacker adapter> --target qwen (the frozen base); the
auditor receives each hidden item's outline, so items stay paired with the base reference.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from curation.quality_rollout import (
    EVALUATOR_CONFIG,
    EVALUATOR_MODEL,
    FAIL_ON_ERROR,
    LOG_SUBDIR,
    MANIFEST_NAME,
    MAX_TURNS,
    REALISM_FILTER,
    RETRY_ATTEMPTS,
    TARGET_CONFIG,
    TARGET_SERVED_MODEL_ALIAS,
    _load_backend,
    _prepare_out,
    _write_manifest,
    behavior_content_hash,
    discover_behaviors,
    runtime_snapshot,
)

PLAYERS = ("claude", "qwen")
QWEN_CONFIG: dict[str, Any] = {**TARGET_CONFIG, "max_connections": 48}
QWEN_IDENTITY = "Qwen3.5-9B, developed by the Qwen team at Alibaba"
CLAUDE_IDENTITY = "Claude Opus 4.7, developed by Anthropic"


def qwen_model(served_name: str) -> str:
    return f"vllm/{served_name}"


def _spec(player: str, claude_connections: int, served_name: str) -> dict[str, Any]:
    if player == "qwen":
        return {"model": qwen_model(served_name), "generate_config": dict(QWEN_CONFIG)}
    return {"model": EVALUATOR_MODEL, "generate_config": {**EVALUATOR_CONFIG, "max_connections": claude_connections}}


def role_specs(
    auditor: str, target: str, claude_connections: int,
    auditor_model: str = TARGET_SERVED_MODEL_ALIAS, target_model: str = TARGET_SERVED_MODEL_ALIAS,
) -> dict[str, dict[str, Any]]:
    claude = _spec("claude", claude_connections, "")
    return {
        "auditor": _spec(auditor, claude_connections, auditor_model),
        "target": _spec(target, claude_connections, target_model),
        "judge": claude,
        "realism": claude,
    }


def build_roles(
    backend: Any, env: Mapping[str, str], auditor: str, target: str, claude_connections: int,
    auditor_model: str = TARGET_SERVED_MODEL_ALIAS, target_model: str = TARGET_SERVED_MODEL_ALIAS,
) -> dict[str, Any]:
    claude = backend.get_model(
        EVALUATOR_MODEL, config=backend.GenerateConfig(**{**EVALUATOR_CONFIG, "max_connections": claude_connections})
    )

    def qwen(url_var: str, served_name: str) -> Any:
        return backend.get_model(
            qwen_model(served_name),
            base_url=env[url_var],
            api_key=env.get("VLLM_API_KEY") or "EMPTY",
            config=backend.GenerateConfig(**QWEN_CONFIG),
        )

    return {
        "auditor": qwen("AUDITOR_BASE_URL", auditor_model) if auditor == "qwen" else claude,
        "target": qwen("TARGET_BASE_URL", target_model) if target == "qwen" else claude,
        "judge": claude,
        "realism": claude,
    }


def run_eval(
    *,
    behaviors_root: Path,
    out: Path,
    auditor: str,
    target: str,
    env: Mapping[str, str] | None = None,
    backend: Any = None,
    claude_connections: int = EVALUATOR_CONFIG["max_connections"],
    auditor_model: str = TARGET_SERVED_MODEL_ALIAS,
    target_model: str = TARGET_SERVED_MODEL_ALIAS,
    limit: int | None = None,
    resume: bool = False,
) -> tuple[bool, list[Any]]:
    env = os.environ if env is None else env
    if auditor not in PLAYERS or target not in PLAYERS:
        raise ValueError(f"auditor and target must be in {PLAYERS}")
    required = ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"]
    required += ["AUDITOR_BASE_URL"] if auditor == "qwen" else []
    required += ["TARGET_BASE_URL"] if target == "qwen" else []
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise SystemExit(f"missing required environment variables: {', '.join(missing)}")
    out = Path(out)
    behaviors = discover_behaviors(Path(behaviors_root))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "judge_evaluation",
        "behaviors": [{"path": str(p), "content_sha256": behavior_content_hash(p)} for p in behaviors],
        "max_turns": MAX_TURNS,
        "realism_filter": REALISM_FILTER,
        "limit": limit,
        "eval_set": {"retry_attempts": RETRY_ATTEMPTS, "fail_on_error": FAIL_ON_ERROR},
        "models": role_specs(auditor, target, claude_connections, auditor_model, target_model),
        "runtime": runtime_snapshot(),
    }
    keep_manifest = _prepare_out(out, manifest, resume)

    backend = backend or _load_backend()
    roles = build_roles(backend, env, auditor, target, claude_connections, auditor_model, target_model)
    tasks = [
        backend.task_with(
            backend.bloom_audit(path, max_turns=MAX_TURNS, realism_filter=REALISM_FILTER),
            name=f"bloom_{path.name}",
            model_roles=roles,
        )
        for path in behaviors
    ]
    if not keep_manifest:
        _write_manifest(out / MANIFEST_NAME, manifest)
    return backend.eval_set(
        tasks,
        log_dir=str(out / LOG_SUBDIR),
        model=roles["target"],
        max_tasks=len(tasks),
        retry_attempts=RETRY_ATTEMPTS,
        fail_on_error=FAIL_ON_ERROR,
        limit=limit,
    )


def rewrite_identity(src: Path, dst: Path) -> int:
    """Copy a behavior root with the target identity changed from Qwen to Claude; returns files rewritten."""
    if dst.exists():
        raise FileExistsError(f"{dst} exists; choose a fresh path")
    shutil.copytree(src, dst, symlinks=False)
    n = 0
    for path in sorted(dst.glob("*/BEHAVIOR.md")):
        text = path.read_text(encoding="utf-8")
        if QWEN_IDENTITY in text:
            path.write_text(text.replace(QWEN_IDENTITY, CLAUDE_IDENTITY), encoding="utf-8")
            n += 1
    return n


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--behaviors", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--auditor", choices=PLAYERS, required=True)
    r.add_argument("--target", choices=PLAYERS, required=True)
    r.add_argument("--claude-connections", type=int, default=EVALUATOR_CONFIG["max_connections"])
    r.add_argument("--auditor-model", default=TARGET_SERVED_MODEL_ALIAS, help="vLLM served name of a Qwen auditor")
    r.add_argument("--target-model", default=TARGET_SERVED_MODEL_ALIAS, help="vLLM served name of a Qwen target")
    r.add_argument("--limit", type=int)
    r.add_argument("--resume", action="store_true")
    i = sub.add_parser("identity")
    i.add_argument("--src", type=Path, required=True)
    i.add_argument("--dst", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "identity":
        print(f"rewrote {rewrite_identity(args.src, args.dst)} BEHAVIOR.md files")
        return 0
    success, _ = run_eval(
        behaviors_root=args.behaviors,
        out=args.out,
        auditor=args.auditor,
        target=args.target,
        claude_connections=args.claude_connections,
        auditor_model=args.auditor_model,
        target_model=args.target_model,
        limit=args.limit,
        resume=args.resume,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
