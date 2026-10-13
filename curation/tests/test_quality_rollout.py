"""Tests for the official Bloom quality rollout wiring (no network, no provider SDKs)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import curation.quality_rollout as qr

ENV = {
    "ANTHROPIC_BASE_URL": "https://anthropic.example.invalid",
    "ANTHROPIC_API_KEY": "sk-test-anthropic",
    "VLLM_BASE_URL": "http://vllm.example.invalid/v1",
    "VLLM_API_KEY": "sk-test-vllm",
}


class FakeConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeBackend:
    """Records every call instead of touching providers or Inspect."""

    def __init__(self):
        self.models = []
        self.audits = []
        self.task_withs = []
        self.eval_set_calls = []
        self.on_eval_set = None
        self.GenerateConfig = FakeConfig

    def get_model(self, name, **kw):
        m = SimpleNamespace(name=name, kw=kw)
        self.models.append(m)
        return m

    def bloom_audit(self, behavior_dir, **kw):
        t = SimpleNamespace(behavior_dir=Path(behavior_dir), kw=kw)
        self.audits.append(t)
        return t

    def task_with(self, task, **kw):
        t = SimpleNamespace(inner=task, kw=kw)
        self.task_withs.append(t)
        return t

    def eval_set(self, tasks, **kw):
        if self.on_eval_set:
            self.on_eval_set(tasks, kw)
        self.eval_set_calls.append((tasks, kw))
        return True, []


def _behaviors(root: Path, names=("alpha", "beta")) -> Path:
    for n in names:
        (root / n).mkdir(parents=True)
        (root / n / "BEHAVIOR.md").write_text("---\nname: x\n---\n")
    (root / "not_a_behavior").mkdir()
    return root


def _run(tmp_path, backend, **kw):
    root = _behaviors(tmp_path / "pilot")
    out = tmp_path / "out"
    args = dict(behaviors_root=root, out=out, env=ENV, backend=backend)
    args.update(kw)
    return root, out, qr.run_quality_rollout(**args)


def test_role_wiring_and_generation_configs(tmp_path):
    be = FakeBackend()
    _run(tmp_path, be)
    roles = be.task_withs[0].kw["model_roles"]
    assert set(roles) == {"auditor", "judge", "realism", "target"}
    for r in ("auditor", "judge", "realism"):
        assert roles[r].name == "anthropic/claude-opus-4-7"
        assert roles[r].kw["config"].reasoning_effort == "medium"
        assert set(roles[r].kw) == {"config"}
        assert ENV["ANTHROPIC_BASE_URL"] not in repr(roles[r].kw)
        assert ENV["ANTHROPIC_API_KEY"] not in repr(roles[r].kw)
    t = roles["target"]
    assert t.name == "vllm/qwen3.5-9b"
    assert t.kw["base_url"] == ENV["VLLM_BASE_URL"]
    assert t.kw["api_key"] == ENV["VLLM_API_KEY"]
    cfg = t.kw["config"]
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (1.0, 0.95, 20)
    assert cfg.presence_penalty == 1.5
    assert cfg.max_tokens == 16384


def test_task_construction_uses_official_defaults(tmp_path):
    be = FakeBackend()
    root, _, _ = _run(tmp_path, be)
    assert [a.behavior_dir for a in be.audits] == [root / "alpha", root / "beta"]
    for a in be.audits:
        assert a.kw == {"max_turns": 5, "realism_filter": 0.6}
    assert [t.kw["name"] for t in be.task_withs] == ["bloom_alpha", "bloom_beta"]
    assert all(t.kw["model_roles"] is be.task_withs[0].kw["model_roles"] for t in be.task_withs)


def test_eval_set_retry_policy_and_target_model(tmp_path):
    be = FakeBackend()
    _, out, _ = _run(tmp_path, be)
    tasks, kw = be.eval_set_calls[0]
    assert len(tasks) == 2
    assert kw["retry_attempts"] == 3
    assert kw["fail_on_error"] == 0.2
    assert kw["log_dir"] == str(out / "logs")
    assert kw["model"] is tasks[0].kw["model_roles"]["target"]
    assert kw["limit"] is None


def test_limit_passthrough(tmp_path):
    be = FakeBackend()
    _run(tmp_path, be, limit=1)
    assert be.eval_set_calls[0][1]["limit"] == 1


def test_manifest_written_before_eval_and_complete(tmp_path):
    be = FakeBackend()
    seen = {}

    def check(tasks, kw):
        seen["manifest"] = json.loads((tmp_path / "out" / "config_manifest.json").read_text())

    be.on_eval_set = check
    root, out, _ = _run(tmp_path, be, limit=1)
    m = seen["manifest"]
    assert m == json.loads((out / "config_manifest.json").read_text())
    assert [item["path"] for item in m["behaviors"]] == [str(root / "alpha"), str(root / "beta")]
    assert all(len(item["content_sha256"]) == 64 for item in m["behaviors"])
    assert m["max_turns"] == 5 and m["realism_filter"] == 0.6 and m["limit"] == 1
    assert m["eval_set"] == {"retry_attempts": 3, "fail_on_error": 0.2}
    assert m["models"]["auditor"]["model"] == "anthropic/claude-opus-4-7"
    assert m["models"]["auditor"]["generate_config"]["reasoning_effort"] == "medium"
    assert m["models"]["target"]["model"] == "vllm/qwen3.5-9b"
    assert m["models"]["target"]["generate_config"] == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
        "max_tokens": 16384, "max_connections": 32,
    }
    assert m["runtime"]["python_version"]
    assert isinstance(m["runtime"]["installed_distributions"], dict)
    assert list(m["runtime"]["installed_distributions"]) == sorted(m["runtime"]["installed_distributions"])
    assert m["runtime"]["repo_git_commit"] is None or len(m["runtime"]["repo_git_commit"]) == 40
    assert m["runtime"]["target_served_model_alias"] == "qwen3.5-9b"


def test_manifest_contains_no_credentials_or_endpoints(tmp_path):
    be = FakeBackend()
    _, out, _ = _run(tmp_path, be)
    text = (out / "config_manifest.json").read_text()
    for v in ENV.values():
        assert v not in text


def test_refuses_non_empty_output_dir_without_calling_anything(tmp_path):
    be = FakeBackend()
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.eval").write_text("x")
    root = _behaviors(tmp_path / "pilot")
    with pytest.raises(FileExistsError):
        qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be)
    assert not be.eval_set_calls and not be.models
    assert (out / "stale.eval").read_text() == "x"


def test_empty_existing_output_dir_is_allowed(tmp_path):
    be = FakeBackend()
    (tmp_path / "out").mkdir()
    _run(tmp_path, be)
    assert len(be.eval_set_calls) == 1


def test_resume_requires_matching_manifest(tmp_path):
    be = FakeBackend()
    root, out, _ = _run(tmp_path, be, limit=1)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / "a.eval").write_text("partial")
    # same config: allowed
    qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be, limit=1, resume=True)
    assert len(be.eval_set_calls) == 2
    # changed config: refused
    with pytest.raises(ValueError, match="manifest"):
        qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be, limit=2, resume=True)
    assert len(be.eval_set_calls) == 2


def test_resume_rejects_changed_behavior_content(tmp_path):
    be = FakeBackend()
    root, out, _ = _run(tmp_path, be, limit=1)
    (root / "alpha" / "nested.txt").write_text("changed after manifest\n")
    with pytest.raises(ValueError, match="manifest"):
        qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be, limit=1, resume=True)
    assert len(be.eval_set_calls) == 1


def test_behavior_hash_does_not_follow_symlink(tmp_path):
    behavior = tmp_path / "behavior"
    behavior.mkdir()
    (behavior / "BEHAVIOR.md").write_text("stable\n")
    external = tmp_path / "external.txt"
    external.write_text("first\n")
    (behavior / "external-link").symlink_to(external)
    before = qr.behavior_content_hash(behavior)
    external.write_text("second\n")
    assert qr.behavior_content_hash(behavior) == before
    (behavior / "external-link").unlink()
    (behavior / "external-link").symlink_to("different-target")
    assert qr.behavior_content_hash(behavior) != before


def test_installed_distribution_snapshot_is_normalized_sorted_and_deterministic(monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": "Z.Pkg"}, version="3"),
        SimpleNamespace(metadata={"Name": "a_pkg"}, version="1"),
        SimpleNamespace(metadata={"Name": "Middle-Pkg"}, version="2"),
    ]
    monkeypatch.setattr(qr.metadata, "distributions", lambda: iter(distributions))
    first = qr.installed_distributions()
    distributions.reverse()
    second = qr.installed_distributions()
    assert first == second == {"a-pkg": "1", "middle-pkg": "2", "z-pkg": "3"}


def test_git_snapshot_is_optional(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(qr.subprocess, "run", unavailable)
    assert qr.repo_git_commit(tmp_path) is None


def test_resume_without_manifest_refused(tmp_path):
    be = FakeBackend()
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk").write_text("x")
    root = _behaviors(tmp_path / "pilot")
    with pytest.raises(ValueError, match="manifest"):
        qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be, resume=True)


def test_fresh_run_recovers_lone_manifest_temp_file(tmp_path):
    be = FakeBackend()
    out = tmp_path / "out"
    out.mkdir()
    temp = out / "config_manifest.json.tmp"
    temp.write_text("interrupted")
    root = _behaviors(tmp_path / "pilot")
    qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be)
    assert not temp.exists()
    assert (out / "config_manifest.json").is_file()
    assert len(be.eval_set_calls) == 1


def test_fresh_run_refuses_manifest_temp_with_any_other_state(tmp_path):
    be = FakeBackend()
    out = tmp_path / "out"
    out.mkdir()
    (out / "config_manifest.json.tmp").write_text("interrupted")
    (out / "other").write_text("keep")
    root = _behaviors(tmp_path / "pilot")
    with pytest.raises(FileExistsError):
        qr.run_quality_rollout(behaviors_root=root, out=out, env=ENV, backend=be)
    assert (out / "config_manifest.json.tmp").read_text() == "interrupted"
    assert (out / "other").read_text() == "keep"
    assert not be.models


def test_no_behaviors_is_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="BEHAVIOR.md"):
        qr.run_quality_rollout(behaviors_root=root, out=tmp_path / "out", env=ENV, backend=FakeBackend())


def test_behavior_root_itself_may_be_a_single_behavior(tmp_path):
    be = FakeBackend()
    root = tmp_path / "one"
    root.mkdir()
    (root / "BEHAVIOR.md").write_text("---\n---\n")
    qr.run_quality_rollout(behaviors_root=root, out=tmp_path / "out", env=ENV, backend=be)
    assert [a.behavior_dir for a in be.audits] == [root]


@pytest.mark.parametrize("missing", ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "VLLM_BASE_URL"])
def test_missing_environment_is_reported(tmp_path, missing):
    be = FakeBackend()
    env = {k: v for k, v in ENV.items() if k != missing}
    root = _behaviors(tmp_path / "pilot")
    with pytest.raises(SystemExit, match=missing):
        qr.run_quality_rollout(behaviors_root=root, out=tmp_path / "out", env=env, backend=be)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_vllm_api_key_defaults_to_empty(tmp_path):
    be = FakeBackend()
    env = {k: v for k, v in ENV.items() if k != "VLLM_API_KEY"}
    _run(tmp_path, be, env=env)
    assert be.task_withs[0].kw["model_roles"]["target"].kw["api_key"] == "EMPTY"


def test_cli_wires_arguments_and_returns_status(tmp_path, monkeypatch):
    be = FakeBackend()
    root = _behaviors(tmp_path / "pilot")
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(qr, "_load_backend", lambda: be)
    rc = qr.main(["--behaviors", str(root), "--out", str(tmp_path / "out"), "--limit", "1"])
    assert rc == 0
    assert be.eval_set_calls[0][1]["limit"] == 1
    assert be.audits[0].kw == {"max_turns": 5, "realism_filter": 0.6}


def test_cli_help_clarifies_limit_is_per_behavior(capsys):
    with pytest.raises(SystemExit) as exc:
        qr.main(["--help"])
    assert exc.value.code == 0
    assert "sample limit per behavior" in capsys.readouterr().out


def test_cli_returns_nonzero_when_eval_set_incomplete(tmp_path, monkeypatch):
    be = FakeBackend()
    be.eval_set = lambda tasks, **kw: (False, [])
    root = _behaviors(tmp_path / "pilot")
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(qr, "_load_backend", lambda: be)
    assert qr.main(["--behaviors", str(root), "--out", str(tmp_path / "out")]) == 1


def test_real_backend_call_signatures_when_available():
    try:
        backend = qr._load_backend()
    except ImportError as exc:
        pytest.skip(f"real rollout dependencies unavailable: {exc}")

    evaluator_config = backend.GenerateConfig(**qr.EVALUATOR_CONFIG)
    target_config = backend.GenerateConfig(**qr.TARGET_CONFIG)
    inspect.signature(backend.get_model).bind(qr.EVALUATOR_MODEL, config=evaluator_config)
    inspect.signature(backend.get_model).bind(
        qr.TARGET_MODEL,
        base_url="http://localhost/v1",
        api_key="test-only",
        config=target_config,
    )
    inspect.signature(backend.bloom_audit).bind(
        Path("."), max_turns=qr.MAX_TURNS, realism_filter=qr.REALISM_FILTER
    )
    inspect.signature(backend.task_with).bind(object(), name="bloom_contract", model_roles={})
    inspect.signature(backend.eval_set).bind(
        [],
        log_dir="logs",
        model=object(),
        max_tasks=1,
        retry_attempts=qr.RETRY_ATTEMPTS,
        fail_on_error=qr.FAIL_ON_ERROR,
        limit=1,
    )
