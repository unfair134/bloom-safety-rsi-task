"""Tests for semantic deduplication with a deterministic fake encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curation.dedupe import (
    choose_threshold,
    edges_at_threshold,
    embed_texts,
    main,
    normalize_text,
    sample_label_pairs,
    similarity_matrix,
)

TOPICS = {
    "rent": np.array([1.0, 0.0, 0.0, 0.0]),
    "vaccine": np.array([0.0, 1.0, 0.0, 0.0]),
    "riddle": np.array([0.0, 0.0, 1.0, 0.0]),
}


def fake_encode(texts: list[str]) -> np.ndarray:
    """Topic one-hot plus a small text-dependent jitter so paraphrases are close but not identical."""
    out = []
    for text in texts:
        topic = next((v for k, v in TOPICS.items() if k in text), np.array([0.0, 0.0, 0.0, 1.0]))
        jitter = (sum(map(ord, text)) % 7) / 100.0
        out.append(topic + np.array([0.0, 0.0, 0.0, jitter]))
    return np.array(out)


def _row(cid: str, behavior: str, text: str, realism: float = 8.0, awareness: float = 1.0) -> dict:
    return {
        "candidate_id": cid,
        "behavior": behavior,
        "scenario_name": cid,
        "seed_text": text,
        "scenario_realism": realism,
        "eval_awareness": awareness,
    }


def _sims(rows: list[dict], tmp_path: Path) -> np.ndarray:
    emb = embed_texts([normalize_text(r["seed_text"]) for r in rows], fake_encode, shard_dir=tmp_path / "emb")
    return similarity_matrix(emb)


def test_normalize_text_collapses_case_width_and_whitespace():
    assert normalize_text("  Ｒent\tIS\n\nthrowing  money ") == "rent is throwing money"


def test_edges_link_duplicates_and_paraphrases_including_cross_behavior(tmp_path: Path):
    rows = [
        _row("a1", "defer", "Rent is throwing money away."),
        _row("a2", "delusion", "rent   is throwing money away."),
        _row("a3", "defer", "Is rent really throwing money away, they ask?"),
        _row("b1", "defer", "The vaccine causes magnetism."),
        _row("c1", "defer", "Pick the best riddle."),
    ]
    edges = edges_at_threshold(_sims(rows, tmp_path), 0.99)
    linked = {frozenset((rows[i]["candidate_id"], rows[j]["candidate_id"])) for i, j, _ in edges}
    assert linked == {frozenset(p) for p in [("a1", "a2"), ("a1", "a3"), ("a2", "a3")]}


def test_choose_threshold_picks_highest_recall_meeting_precision():
    labeled = [(0.95, True), (0.9, True), (0.85, True), (0.8, False), (0.75, True), (0.3, False)]
    result = choose_threshold(labeled, min_precision=0.95)
    assert result["threshold"] == 0.85
    assert result["recall"] == pytest.approx(0.75)
    result = choose_threshold(labeled, min_precision=0.8)
    assert result["threshold"] == 0.75


def test_choose_threshold_requires_duplicates():
    with pytest.raises(ValueError):
        choose_threshold([(0.9, False)])


def test_embedding_shards_resume_and_reject_changed_texts(tmp_path: Path):
    calls: list[int] = []

    def counting(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return fake_encode(texts)

    texts = [f"rent {i}" for i in range(5)]
    first = embed_texts(texts, counting, shard_dir=tmp_path / "e", shard_size=2)
    assert calls == [2, 2, 1]
    second = embed_texts(texts, counting, shard_dir=tmp_path / "e", shard_size=2)
    assert calls == [2, 2, 1]
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        embed_texts(texts + ["new"], counting, shard_dir=tmp_path / "e", shard_size=2)


def test_label_pairs_are_deterministic_and_mixed(tmp_path: Path):
    rows = [_row(f"r{i:02d}", "defer", f"{['rent', 'vaccine', 'riddle', 'other'][i % 4]} {i}") for i in range(20)]
    sims = _sims(rows, tmp_path)
    a = sample_label_pairs(rows, sims, n_nn=5, n_random=5, seed=1)
    b = sample_label_pairs(rows, sims, n_nn=5, n_random=5, seed=1)
    assert a == b
    assert {p["source"] for p in a} == {"nearest_neighbor", "random"}
    assert len({(min(p["a"], p["b"]), max(p["a"], p["b"])) for p in a}) == len(a)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    texts = {
        "id1": ("defer", "Rent is throwing money away."),
        "id2": ("defer", "rent is throwing money away."),
        "id3": ("delusion", "The vaccine causes magnetism."),
        "id4": ("political", "Pick the best riddle."),
        "id5": ("political", "Something unrelated."),
    }
    inventory = tmp_path / "inventory.jsonl"
    quality = tmp_path / "quality.jsonl"
    inventory.write_text("".join(json.dumps({"id": k, "seed_text": t}) + "\n" for k, (_, t) in texts.items()))
    q_rows = [
        {
            "candidate_id": k,
            "behavior": b,
            "scenario_name": k,
            "accepted": k != "id5",
            "official_scores": {"scenario_realism": 8.0, "eval_awareness": 1.0, "behavior_score": 5.0},
        }
        for k, (b, _) in texts.items()
    ]
    quality.write_text("".join(json.dumps(r) + "\n" for r in q_rows))
    return quality, inventory


def test_cli_pairs_threshold_cluster_end_to_end(tmp_path: Path):
    quality, inventory = _write_inputs(tmp_path)
    out = tmp_path / "dedupe"
    common = ["--quality", str(quality), "--inventory", str(inventory), "--out", str(out)]
    assert main(["pairs", *common, "--n-nn", "3", "--n-random", "2"], encode=fake_encode) == 0
    key = [json.loads(line) for line in (out / "pairs_key.jsonl").read_text().splitlines()]
    blind = [json.loads(line) for line in (out / "pairs_blind.jsonl").read_text().splitlines()]
    assert all(set(row) == {"pair_id", "text_a", "text_b"} for row in blind)
    assert all("id5" not in (row["a"], row["b"]) for row in key)

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps({"pair_id": r["pair_id"], "duplicate": {r["a"], r["b"]} == {"id1", "id2"}}) + "\n" for r in key)
    )
    threshold = tmp_path / "threshold.json"
    assert main(["threshold", "--key", str(out / "pairs_key.jsonl"), "--labels", str(labels), "--out", str(threshold)]) == 0

    assert main(["edges", *common, "--threshold-file", str(threshold)], encode=fake_encode) == 0
    manifest = json.loads((out / "edges_manifest.json").read_text())
    assert manifest["accepted"] == 4
    assert manifest["edges"] == 1
    roster = [json.loads(line) for line in (out / "roster.jsonl").read_text().splitlines()]
    assert sorted(r["candidate_id"] for r in roster) == ["id1", "id2", "id3", "id4"]
    edges = [json.loads(line) for line in (out / "edges.jsonl").read_text().splitlines()]
    assert {edges[0]["a"], edges[0]["b"]} == {"id1", "id2"}


def test_threshold_refuses_unlabeled_pairs(tmp_path: Path):
    key = tmp_path / "key.jsonl"
    key.write_text(json.dumps({"pair_id": "pair-000", "cosine": 0.9}) + "\n")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("")
    assert main(["threshold", "--key", str(key), "--labels", str(labels), "--out", str(tmp_path / "t.json")]) == 1
