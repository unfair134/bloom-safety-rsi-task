"""Tests for frozen partitions with leakage-free held-out sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from curation.split import assign_partitions, main, verify_partitions


def _roster(counts: dict[str, int]) -> list[dict]:
    return [{"candidate_id": f"{b}-{i:04d}", "behavior": b} for b, n in counts.items() for i in range(n)]


def _per(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["behavior"]] = out.get(r["behavior"], 0) + 1
    return out


def _home(parts: dict) -> dict[str, str]:
    return {r["candidate_id"]: name for name, rows in parts.items() for r in rows}


def test_exact_quotas_with_balanced_seed():
    parts = assign_partitions(_roster({"a": 400, "b": 400, "c": 200}), [], test_per_behavior=10, val_per_behavior=5, seed_total=300)
    assert _per(parts["final_test"]) == {"a": 10, "b": 10, "c": 10}
    assert _per(parts["feedback_validation"]) == {"a": 5, "b": 5, "c": 5}
    assert _per(parts["seed"]) == {"a": 100, "b": 100, "c": 100}


def test_seed_shortfall_redistributed_to_other_behaviors():
    parts = assign_partitions(_roster({"a": 300, "b": 300, "small": 20}), [], test_per_behavior=10, val_per_behavior=5, seed_total=300)
    seed = _per(parts["seed"])
    assert seed["small"] == 5
    assert sum(seed.values()) == 300
    assert abs(seed["a"] - seed["b"]) <= 1


def test_duplicates_inside_a_partition_are_kept():
    roster = _roster({"a": 50})
    one = assign_partitions(roster, [], test_per_behavior=10, val_per_behavior=5, seed_total=20)
    test_ids = [r["candidate_id"] for r in one["final_test"]]
    edges = [{"a": test_ids[0], "b": test_ids[1], "cosine": 0.99}]
    two = assign_partitions(roster, edges, test_per_behavior=10, val_per_behavior=5, seed_total=20)
    assert [r["candidate_id"] for r in two["final_test"]] == test_ids


def test_neighbors_of_held_out_never_enter_seed_or_validation_even_across_behaviors():
    roster = _roster({"a": 60, "b": 60})
    base = assign_partitions(roster, [], test_per_behavior=10, val_per_behavior=5, seed_total=50)
    t = base["final_test"][0]["candidate_id"]
    v = base["feedback_validation"][0]["candidate_id"]
    seeds = [r["candidate_id"] for r in base["seed"]]
    edges = [
        {"a": t, "b": base["feedback_validation"][-1]["candidate_id"], "cosine": 0.97},
        {"a": t, "b": seeds[0], "cosine": 0.97},
        {"a": v, "b": seeds[-1], "cosine": 0.97},
    ]
    parts = assign_partitions(roster, edges, test_per_behavior=10, val_per_behavior=5, seed_total=50)
    verify_partitions(parts, edges)
    home = _home(parts)
    assert home[base["feedback_validation"][-1]["candidate_id"]] != "feedback_validation"
    assert home[seeds[0]] == "reserve"
    assert _per(parts["feedback_validation"]) == {"a": 5, "b": 5}
    reasons = {r["reserve_reason"] for r in parts["reserve"]}
    assert "near_held_out" in reasons


def test_split_is_deterministic_and_seed_dependent():
    roster = _roster({"a": 200, "b": 200})
    one = assign_partitions(roster, [], test_per_behavior=10, val_per_behavior=5, seed_total=100, split_seed=0)
    two = assign_partitions(list(reversed(roster)), [], test_per_behavior=10, val_per_behavior=5, seed_total=100, split_seed=0)
    other = assign_partitions(roster, [], test_per_behavior=10, val_per_behavior=5, seed_total=100, split_seed=1)
    assert one == two
    assert one["final_test"] != other["final_test"]


def test_fails_when_a_behavior_is_too_small_for_held_out():
    with pytest.raises(ValueError, match="small: 8 candidates"):
        assign_partitions(_roster({"a": 100, "small": 8}), [], test_per_behavior=10, val_per_behavior=5, seed_total=10)
    with pytest.raises(ValueError, match="small: 2 eligible"):
        assign_partitions(_roster({"a": 100, "small": 12}), [], test_per_behavior=10, val_per_behavior=5, seed_total=10)


def test_fails_when_seed_cannot_be_filled():
    with pytest.raises(ValueError, match="remain for seed"):
        assign_partitions(_roster({"a": 20, "b": 20}), [], test_per_behavior=10, val_per_behavior=5, seed_total=20)


def test_verify_detects_held_out_leak_but_allows_within_partition_and_seed_pairs():
    parts = {
        "final_test": [{"candidate_id": "t1"}, {"candidate_id": "t2"}],
        "feedback_validation": [{"candidate_id": "v1"}],
        "seed": [{"candidate_id": "s1"}, {"candidate_id": "s2"}],
        "reserve": [{"candidate_id": "r1"}],
    }
    verify_partitions(parts, [{"a": "t1", "b": "t2"}, {"a": "s1", "b": "s2"}, {"a": "t1", "b": "r1"}])
    with pytest.raises(ValueError, match="cross a held-out"):
        verify_partitions(parts, [{"a": "v1", "b": "s1"}])
    with pytest.raises(ValueError, match="cross a held-out"):
        verify_partitions(parts, [{"a": "t1", "b": "v1"}])


def test_cli_writes_frozen_manifests_and_refuses_overwrite(tmp_path: Path):
    roster = _roster({"a": 40, "b": 40})
    rpath = tmp_path / "roster.jsonl"
    rpath.write_text("".join(json.dumps(r) + "\n" for r in roster))
    epath = tmp_path / "edges.jsonl"
    epath.write_text(json.dumps({"a": "a-0000", "b": "b-0000", "cosine": 0.99}) + "\n")
    ipath = tmp_path / "inventory.jsonl"
    ipath.write_text("".join(json.dumps({"id": r["candidate_id"], "seed_path": f"p/{r['candidate_id']}.md", "seed_text": "t"}) + "\n" for r in roster))
    out = tmp_path / "split"
    args = ["--roster", str(rpath), "--edges", str(epath), "--inventory", str(ipath), "--out", str(out),
            "--test-per-behavior", "5", "--val-per-behavior", "3", "--seed-total", "30"]
    assert main(args) == 0
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["partitions"]["final_test"]["per_behavior"] == {"a": 5, "b": 5}
    assert manifest["partitions"]["seed"]["count"] == 30
    assert sum(p["count"] for p in manifest["partitions"].values()) == len(roster)
    ids = [json.loads(l)["candidate_id"] for name in ("seed", "final_test", "feedback_validation", "reserve") for l in (out / f"{name}.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids))
    assert main(args) == 1
