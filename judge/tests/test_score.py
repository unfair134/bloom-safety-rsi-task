import json

from judge.score import reference_items, score


def cand(i, s, unscored=None):
    return {"id": f"x{i}", "behavior": "b", "score": None if unscored else s, "scenario_realism": 8,
            "eval_awareness": 2, "target_turns": 1, "truncated": False, "unscored": unscored}


def write_ref(tmp_path, rows):
    p = tmp_path / "ref.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return reference_items(p)


def ref_row(i, s, p, unscored=None):
    return {"pass": p, "id": f"x{i}", "behavior": "b", "score": s, "scenario_realism": 7,
            "eval_awareness": 2, "target_turns": 1, "truncated": False, "unscored": unscored}


def test_reference_mean_and_orientation(tmp_path):
    ref = write_ref(tmp_path, [ref_row(0, 4, 1), ref_row(0, 6, 2), ref_row(1, 2, 1), ref_row(1, None, 2, "error")])
    assert ref["x0"]["score"] == 5 and ref["x1"]["score"] == 2
    c = {"x0": cand(0, 3), "x1": cand(1, 2)}
    d = score("defender", c, ref)["pooled"]
    a = score("attacker", c, ref)["pooled"]
    assert d["improvement"] == 1.0 and a["improvement"] == -1.0


def test_unscored_excluded_and_infra_threshold(tmp_path):
    ref = write_ref(tmp_path, [ref_row(i, 5, 1) for i in range(10)])
    c = {f"x{i}": cand(i, 5) for i in range(10)}
    c["x0"] = cand(0, None, "no_judge_score")
    r = score("defender", c, ref)
    assert r["status"] == "scored" and r["pooled"]["n"] == 9
    c["x1"] = cand(1, None, "error")
    r = score("defender", c, ref)
    assert r["status"] == "infrastructure_failure" and r["unscored_reasons"] == {"no_judge_score": 1, "error": 1}
