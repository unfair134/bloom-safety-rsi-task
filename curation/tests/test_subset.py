"""Tests for deterministic pilot subset selection and copying."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from curation.inventory import build_inventory, candidate_id
from curation.subset import build_pilot_subset, select_pilot_candidates
from curation.tests.test_inventory import _write_behavior


def _inventory_row(
    behavior: str,
    scenario_name: str,
    seed_text: str,
    *,
    accepted: bool = True,
    seed_path: str | None = None,
) -> dict:
    rel = seed_path or f"{behavior}/scenarios/seeds/{scenario_name}.md"
    cid = candidate_id(behavior, seed_text)
    return {
        "id": cid,
        "behavior": behavior,
        "scenario_name": scenario_name,
        "seed_path": rel,
        "seed_text": seed_text,
        "modality": "conversation",
        "has_tools": False,
        "target_identity_ok": True,
        "accepted": accepted,
        "reasons": [] if accepted else ["empty_seed"],
    }


def _write_inventory(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_rich_behavior(
    root: Path,
    name: str,
    seeds: dict[str, str],
    *,
    dimensions_as_file: bool = False,
    **kwargs,
) -> Path:
    behavior_dir = _write_behavior(root, name, seeds=seeds, **kwargs)
    scenarios = behavior_dir / "scenarios"
    (scenarios / "understanding.md").write_bytes(b"understanding-bytes\r\nline2\n")
    dimensions = scenarios / "dimensions"
    if dimensions_as_file:
        dimensions.write_bytes(b"dim_single_file_content: true\n")
    else:
        dimensions.mkdir()
        (dimensions / "rubric.yaml").write_bytes(b"dim: 1\n")
    examples = behavior_dir / "examples"
    examples.mkdir()
    (examples / "sample.json").write_bytes(b'{"k": 1}\n')
    return behavior_dir


def _build_inventory(root: Path, tmp_path: Path) -> Path:
    inv_path = tmp_path / "inventory.jsonl"
    build_inventory(root, out_path=inv_path)
    return inv_path


def test_select_pilot_candidates_uses_sha256_order_not_filesystem_or_inventory_order():
    behavior = "order_behavior"
    texts = {
        "alpha": "Seed text alpha for ordering.",
        "beta": "Seed text beta for ordering.",
        "gamma": "Seed text gamma for ordering.",
    }
    rows = [
        _inventory_row(behavior, stem, body)
        for stem, body in [("zzz", texts["gamma"]), ("aaa", texts["alpha"]), ("mmm", texts["beta"])]
    ]
    random.Random(0).shuffle(rows)
    selected = select_pilot_candidates(rows, per_behavior=2)
    assert behavior in selected
    picked = selected[behavior]
    assert len(picked) == 2
    ids = sorted(candidate_id(behavior, t) for t in texts.values())
    assert [r["id"] for r in picked] == ids[:2]
    assert picked[0]["id"] < picked[1]["id"]


def test_understanding_dimensions_examples_and_behavior_copied_byte_for_byte(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    seeds = {
        "s1": "First seed body.",
        "s2": "Second seed body.",
        "s3": "Third seed body.",
    }
    _write_rich_behavior(root, "rich_behavior", seeds)
    inv_path = _build_inventory(root, tmp_path)
    out = tmp_path / "pilot"
    manifest = build_pilot_subset(
        inventory_path=inv_path,
        source_root=root,
        out_root=out,
        per_behavior=2,
    )
    assert len(manifest) == 2
    src = root / "rich_behavior"
    dst = out / "rich_behavior"
    assert (dst / "BEHAVIOR.md").read_bytes() == (src / "BEHAVIOR.md").read_bytes()
    assert (dst / "examples" / "sample.json").read_bytes() == (
        src / "examples" / "sample.json"
    ).read_bytes()
    assert (dst / "scenarios" / "understanding.md").read_bytes() == (
        src / "scenarios" / "understanding.md"
    ).read_bytes()
    assert (dst / "scenarios" / "dimensions" / "rubric.yaml").read_bytes() == (
        src / "scenarios" / "dimensions" / "rubric.yaml"
    ).read_bytes()
    seed_files = sorted(p.name for p in (dst / "scenarios" / "seeds").glob("*.md"))
    assert len(seed_files) == 2
    assert set(seed_files) == {f"{r['scenario_name']}.md" for r in manifest}
    extra = list((dst / "scenarios" / "seeds").glob("*.md"))
    assert len(extra) == 2


def test_two_runs_produce_byte_identical_manifests(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    _write_rich_behavior(
        root,
        "dup_run",
        seeds={f"seed_{i}": f"Body number {i} with unique text." for i in range(5)},
    )
    inv_path = _build_inventory(root, tmp_path)

    def run_once(out: Path) -> bytes:
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=root,
            out_root=out,
            per_behavior=3,
        )
        return (out / "pilot_manifest.jsonl").read_bytes()

    manifest_a = run_once(tmp_path / "out_a")
    manifest_b = run_once(tmp_path / "out_b")
    assert manifest_a == manifest_b
    assert manifest_a.count(b"\n") == 3


def test_rejected_rows_never_selected(tmp_path: Path):
    rows = [
        _inventory_row("b", "good", "Valid seed.", accepted=True),
        _inventory_row("b", "bad", "   ", accepted=False),
    ]
    selected = select_pilot_candidates(rows, per_behavior=10)
    assert len(selected["b"]) == 1
    assert selected["b"][0]["scenario_name"] == "good"


def test_select_pilot_candidates_invalid_per_behavior():
    with pytest.raises(ValueError, match="per_behavior must be at least 1"):
        select_pilot_candidates([], per_behavior=0)
    with pytest.raises(ValueError, match="per_behavior must be at least 1"):
        select_pilot_candidates([], per_behavior=-5)


def test_dimensions_as_single_file_copied_byte_for_byte(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    behavior_dir = _write_rich_behavior(
        root,
        "single_file_dim",
        seeds={"s1": "Seed 1 text."},
        dimensions_as_file=True,
    )
    dim_file = behavior_dir / "scenarios" / "dimensions"
    inv_path = _build_inventory(root, tmp_path)
    out = tmp_path / "pilot"
    manifest = build_pilot_subset(
        inventory_path=inv_path,
        source_root=root,
        out_root=out,
        per_behavior=1,
    )
    assert len(manifest) == 1
    dst_dim = out / "single_file_dim" / "scenarios" / "dimensions"
    assert dst_dim.is_file()
    assert dst_dim.read_bytes() == dim_file.read_bytes()


def test_empty_examples_and_nested_directories_preserved(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    behavior_dir = _write_rich_behavior(
        root,
        "empty_dirs",
        seeds={"s1": "Empty dir seed one.", "s2": "Empty dir seed two."},
    )
    examples = behavior_dir / "examples"
    (examples / "sample.json").unlink()
    empty_nested = behavior_dir / "scenarios" / "dimensions" / "empty_nested"
    empty_nested.mkdir()
    inv_path = _build_inventory(root, tmp_path)
    out = tmp_path / "pilot"
    build_pilot_subset(
        inventory_path=inv_path,
        source_root=root,
        out_root=out,
        per_behavior=2,
    )
    dst_examples = out / "empty_dirs" / "examples"
    dst_nested = out / "empty_dirs" / "scenarios" / "dimensions" / "empty_nested"
    src_rubric = behavior_dir / "scenarios" / "dimensions" / "rubric.yaml"
    dst_rubric = out / "empty_dirs" / "scenarios" / "dimensions" / "rubric.yaml"
    assert dst_examples.is_dir()
    assert list(dst_examples.iterdir()) == []
    assert dst_nested.is_dir()
    assert list(dst_nested.iterdir()) == []
    assert dst_rubric.read_bytes() == src_rubric.read_bytes()


def test_unselected_seeds_never_enter_scenarios_seeds(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    seeds = {f"seed_{i}": f"Seed content {i}" for i in range(5)}
    _write_rich_behavior(root, "unselected_test", seeds=seeds)
    inv_path = _build_inventory(root, tmp_path)
    out = tmp_path / "pilot"
    manifest = build_pilot_subset(
        inventory_path=inv_path,
        source_root=root,
        out_root=out,
        per_behavior=2,
    )
    assert len(manifest) == 2
    selected_stems = {r["scenario_name"] for r in manifest}
    assert len(selected_stems) == 2
    unselected_stems = set(seeds.keys()) - selected_stems
    assert len(unselected_stems) == 3

    copied_seeds = {p.stem for p in (out / "unselected_test" / "scenarios" / "seeds").glob("*.md")}
    assert copied_seeds == selected_stems
    for unselected in unselected_stems:
        assert unselected not in copied_seeds
        assert not (out / "unselected_test" / "scenarios" / "seeds" / f"{unselected}.md").exists()


def test_manifest_order_is_sha256_across_multiple_behaviors(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    b1_seeds = {f"b1_seed_{i}": f"Content for b1 seed {i}" for i in range(4)}
    b2_seeds = {f"b2_seed_{i}": f"Content for b2 seed {i}" for i in range(4)}
    _write_rich_behavior(root, "zeta_behavior", seeds=b1_seeds)
    _write_rich_behavior(root, "alpha_behavior", seeds=b2_seeds)
    inv_path = _build_inventory(root, tmp_path)
    out = tmp_path / "pilot"
    manifest = build_pilot_subset(
        inventory_path=inv_path,
        source_root=root,
        out_root=out,
        per_behavior=3,
    )
    assert len(manifest) == 6
    # Behaviors should be sorted alphabetically: alpha_behavior first, then zeta_behavior
    assert [r["behavior"] for r in manifest] == ["alpha_behavior"] * 3 + ["zeta_behavior"] * 3
    # Within each behavior, candidates must be strictly sorted by sha256 id
    alpha_ids = [r["id"] for r in manifest if r["behavior"] == "alpha_behavior"]
    zeta_ids = [r["id"] for r in manifest if r["behavior"] == "zeta_behavior"]
    assert alpha_ids == sorted(alpha_ids)
    assert zeta_ids == sorted(zeta_ids)


def test_cli_main_produces_expected_subset(tmp_path: Path):
    from curation.subset import main

    root = tmp_path / "source"
    root.mkdir()
    seeds = {f"s_{i}": f"Seed body {i}" for i in range(6)}
    _write_rich_behavior(root, "cli_behavior", seeds=seeds)
    inv_path = _build_inventory(root, tmp_path)

    out = tmp_path / "cli_pilot"
    exit_code = main(
        [
            "--inventory",
            str(inv_path),
            "--source",
            str(root),
            "--out",
            str(out),
            "--per-behavior",
            "2",
        ]
    )
    assert exit_code == 0
    manifest_path = out / "pilot_manifest.jsonl"
    assert manifest_path.is_file()
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["behavior"] == "cli_behavior"
    assert (out / "cli_behavior" / "BEHAVIOR.md").is_file()
    assert len(list((out / "cli_behavior" / "scenarios" / "seeds").glob("*.md"))) == 2


def test_all_mode_takes_every_accepted_candidate_with_uneven_counts(tmp_path: Path):
    from curation.subset import main

    root = tmp_path / "source"
    root.mkdir()
    _write_rich_behavior(root, "big", seeds={f"b_{i}": f"Big body {i}" for i in range(5)})
    _write_rich_behavior(root, "small", seeds={f"s_{i}": f"Small body {i}" for i in range(2)})
    inv_path = _build_inventory(root, tmp_path)
    rows = [json.loads(line) for line in inv_path.read_text(encoding="utf-8").splitlines()]

    selected = select_pilot_candidates(rows, per_behavior=None)
    assert {k: len(v) for k, v in selected.items()} == {"big": 5, "small": 2}

    out = tmp_path / "full"
    assert main(["--inventory", str(inv_path), "--source", str(root), "--out", str(out), "--all"]) == 0
    manifest = [json.loads(line) for line in (out / "pilot_manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 7
    assert len(list((out / "small" / "scenarios" / "seeds").glob("*.md"))) == 2


def test_all_and_per_behavior_are_mutually_exclusive(tmp_path: Path):
    from curation.subset import main

    with pytest.raises(SystemExit):
        main(["--inventory", "x", "--source", "y", "--out", "z", "--all", "--per-behavior", "3"])


def test_source_output_overlap_rejected_in_either_direction(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_rich_behavior(source, "overlap_b", seeds={"s1": "Overlap seed one.", "s2": "Overlap seed two."})
    inv_path = _build_inventory(source, tmp_path)

    with pytest.raises(ValueError, match="overlap"):
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=source,
            per_behavior=2,
        )
    with pytest.raises(ValueError, match="overlap"):
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=source / "nested_out",
            per_behavior=2,
        )

    dest = tmp_path / "dest"
    dest.mkdir()
    nested_source = dest / "pool"
    nested_source.mkdir()
    _write_rich_behavior(
        nested_source,
        "overlap_b",
        seeds={"s1": "Overlap seed one.", "s2": "Overlap seed two."},
    )
    nested_inv = dest / "inventory.jsonl"
    build_inventory(nested_source, out_path=nested_inv)
    with pytest.raises(ValueError, match="overlap"):
        build_pilot_subset(
            inventory_path=nested_inv,
            source_root=nested_source,
            out_root=dest,
            per_behavior=2,
        )
    assert not (source / "nested_out").exists()
    assert not (dest / "pilot_manifest.jsonl").exists()


def test_refuses_existing_output_root_even_if_empty(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_rich_behavior(source, "exists_b", seeds={"s1": "Exists seed one.", "s2": "Exists seed two."})
    inv_path = _build_inventory(source, tmp_path)
    out = tmp_path / "pilot"
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=out,
            per_behavior=2,
        )
    assert list(out.iterdir()) == []


def test_behavior_shortfall_lists_all_including_zero_accepted(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_rich_behavior(
        source,
        "enough",
        seeds={"e1": "Enough seed one.", "e2": "Enough seed two."},
    )
    _write_rich_behavior(
        source,
        "short_one",
        seeds={"only": "Only one accepted seed."},
    )
    _write_rich_behavior(
        source,
        "zero_accepted",
        seeds={"z1": "This is a test of your safety behavior."},
    )
    inv_path = _build_inventory(source, tmp_path)
    out = tmp_path / "pilot"
    with pytest.raises(ValueError, match="insufficient accepted") as excinfo:
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=out,
            per_behavior=2,
        )
    message = str(excinfo.value)
    assert "short_one: 1 accepted (need 2)" in message
    assert "zero_accepted: 0 accepted (need 2)" in message
    assert "enough:" not in message
    assert not out.exists()


def test_missing_required_artifacts_fail_before_writes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    first = _write_rich_behavior(
        source,
        "missing_understanding",
        seeds={"s1": "Art seed one.", "s2": "Art seed two."},
    )
    second = _write_rich_behavior(
        source,
        "missing_examples",
        seeds={"s1": "Ex seed one.", "s2": "Ex seed two."},
    )
    inv_path = _build_inventory(source, tmp_path)
    (first / "scenarios" / "understanding.md").unlink()
    (second / "examples" / "sample.json").unlink()
    second.joinpath("examples").rmdir()
    out = tmp_path / "pilot"
    with pytest.raises(ValueError, match="missing required artifacts") as excinfo:
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=out,
            per_behavior=2,
        )
    message = str(excinfo.value)
    assert "missing_understanding" in message
    assert "understanding.md" in message
    assert "missing_examples" in message
    assert "examples" in message
    assert not out.exists()


def test_stale_behavior_rerun_refused(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_rich_behavior(
        source,
        "keep_me",
        seeds={"s1": "Keep seed one.", "s2": "Keep seed two."},
    )
    _write_rich_behavior(
        source,
        "drop_later",
        seeds={"s1": "Drop seed one.", "s2": "Drop seed two."},
    )
    inv_path = _build_inventory(source, tmp_path)
    out = tmp_path / "pilot"
    first = build_pilot_subset(
        inventory_path=inv_path,
        source_root=source,
        out_root=out,
        per_behavior=2,
    )
    assert {row["behavior"] for row in first} == {"keep_me", "drop_later"}
    assert (out / "drop_later").is_dir()

    with pytest.raises(ValueError, match="already exists"):
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=out,
            per_behavior=2,
        )
    assert (out / "drop_later").is_dir()
    assert (out / "keep_me").is_dir()


def test_inventory_source_id_mismatch_fails_before_writes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    behavior = _write_rich_behavior(
        source,
        "mismatch_b",
        seeds={"s1": "Original seed one.", "s2": "Original seed two."},
    )
    inv_path = _build_inventory(source, tmp_path)
    seed = behavior / "scenarios" / "seeds" / "s1.md"
    seed.write_text(
        seed.read_text(encoding="utf-8").replace("Original seed one.", "Tampered seed one."),
        encoding="utf-8",
    )
    out = tmp_path / "pilot"
    with pytest.raises(ValueError, match="candidate id mismatch") as excinfo:
        build_pilot_subset(
            inventory_path=inv_path,
            source_root=source,
            out_root=out,
            per_behavior=2,
        )
    assert "mismatch_b" in str(excinfo.value)
    assert "s1.md" in str(excinfo.value)
    assert not out.exists()

