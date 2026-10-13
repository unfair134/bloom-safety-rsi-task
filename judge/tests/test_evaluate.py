import json

import pytest

from judge import evaluate
from curation.tests.test_quality_rollout import FakeBackend, _behaviors

CLAUDE_ENV = {"ANTHROPIC_BASE_URL": "http://x", "ANTHROPIC_API_KEY": "k"}
QWEN = "vllm/qwen3.5-9b"
CLAUDE = "anthropic/claude-opus-4-7"


def _roles(tmp_path, auditor, target, env, **kw):
    be = FakeBackend()
    out = tmp_path / "out"
    ok, _ = evaluate.run_eval(behaviors_root=_behaviors(tmp_path / "r"), out=out, auditor=auditor, target=target, env=env, backend=be, **kw)
    assert ok
    assert all(a.kw == {"max_turns": 5, "realism_filter": 0.6} for a in be.audits)
    return be.task_withs[0].kw["model_roles"], json.loads((out / "config_manifest.json").read_text())


def test_qwen_auditor_and_target_use_separate_endpoints(tmp_path):
    env = {**CLAUDE_ENV, "AUDITOR_BASE_URL": "http://a/v1", "TARGET_BASE_URL": "http://t/v1"}
    roles, manifest = _roles(tmp_path, "qwen", "qwen", env)
    assert roles["auditor"].name == roles["target"].name == QWEN
    assert roles["auditor"].kw["base_url"] == "http://a/v1" and roles["target"].kw["base_url"] == "http://t/v1"
    assert roles["judge"].name == roles["realism"].name == CLAUDE
    assert manifest["purpose"] == "judge_evaluation" and manifest["models"]["auditor"]["model"] == QWEN
    assert "http://" not in json.dumps(manifest)


def test_adapter_served_names_per_role(tmp_path):
    env = {**CLAUDE_ENV, "AUDITOR_BASE_URL": "http://a/v1", "TARGET_BASE_URL": "http://t/v1"}
    roles, manifest = _roles(tmp_path, "qwen", "qwen", env, auditor_model="attacker")
    assert roles["auditor"].name == "vllm/attacker" and roles["target"].name == QWEN
    assert manifest["models"]["auditor"]["model"] == "vllm/attacker"
    roles, _ = _roles(tmp_path / "d", "claude", "qwen", {**CLAUDE_ENV, "TARGET_BASE_URL": "http://t/v1"},
                      target_model="defender")
    assert roles["auditor"].name == CLAUDE and roles["target"].name == "vllm/defender"


def test_claude_auditor_and_target(tmp_path):
    roles, manifest = _roles(tmp_path, "claude", "claude", CLAUDE_ENV)
    assert {m.name for m in roles.values()} == {CLAUDE}
    assert manifest["models"]["target"]["model"] == CLAUDE


def test_missing_qwen_endpoint_refused(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.run_eval(
            behaviors_root=_behaviors(tmp_path / "r"), out=tmp_path / "o", auditor="claude", target="qwen", env=CLAUDE_ENV, backend=FakeBackend()
        )


def test_rewrite_identity(tmp_path):
    src = tmp_path / "src" / "b"
    src.mkdir(parents=True)
    (src / "BEHAVIOR.md").write_text(f"x: The model we are evaluating is {evaluate.QWEN_IDENTITY}.\n")
    (src / "seed.md").write_text(evaluate.QWEN_IDENTITY)
    assert evaluate.rewrite_identity(tmp_path / "src", tmp_path / "dst") == 1
    assert evaluate.CLAUDE_IDENTITY in (tmp_path / "dst" / "b" / "BEHAVIOR.md").read_text()
    assert (tmp_path / "dst" / "b" / "seed.md").read_text() == evaluate.QWEN_IDENTITY
    with pytest.raises(FileExistsError):
        evaluate.rewrite_identity(tmp_path / "src", tmp_path / "dst")
