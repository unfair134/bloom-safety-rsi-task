"""Near-duplicate detection used only to keep held-out partitions free of leakage.

The candidate pool is left exactly as petri_bloom generated it: nothing is removed here. The output
is the set of near-duplicate pairs, which ``curation.split`` uses so that no feedback-validation
item is near a final-test item and no seed item is near either.

Three subcommands, run in order:

* ``pairs``: embed accepted seeds and write a blinded set of candidate pairs for labeling
  (nearest-neighbor pairs stratified over similarity rank, plus random controls).
* ``threshold``: from the labels, choose the highest-recall cosine threshold whose duplicate
  precision is at least the target, and freeze it.
* ``edges``: write the accepted roster and every pair at or above the frozen threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

MODEL_MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")

Encoder = Callable[[list[str]], np.ndarray]


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_accepted(quality_path: Path, inventory_path: Path) -> list[dict[str, Any]]:
    """Accepted quality rows joined to inventory seed text, sorted by candidate id."""
    inventory = {row["id"]: row for row in _read_jsonl(inventory_path)}
    rows: list[dict[str, Any]] = []
    for q in _read_jsonl(quality_path):
        if not q.get("accepted"):
            continue
        cid = q["candidate_id"]
        if cid not in inventory:
            raise ValueError(f"accepted candidate {cid} missing from inventory")
        scores = q.get("official_scores") or {}
        rows.append(
            {
                "candidate_id": cid,
                "behavior": q["behavior"],
                "scenario_name": q.get("scenario_name"),
                "seed_text": inventory[cid]["seed_text"],
                "scenario_realism": float(scores["scenario_realism"]),
                "eval_awareness": float(scores["eval_awareness"]),
            }
        )
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids among accepted rows")
    return sorted(rows, key=lambda row: row["candidate_id"])


def embed_texts(
    texts: list[str],
    encode: Encoder,
    *,
    shard_dir: Path,
    shard_size: int = 1000,
) -> np.ndarray:
    """Embed in shards written atomically to ``shard_dir``; existing shards are reused on rerun."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    stamp = shard_dir / "texts.sha256"
    if stamp.exists() and stamp.read_text().strip() != digest:
        raise ValueError(f"{shard_dir} holds embeddings for different texts")
    _atomic_write_text(stamp, digest + "\n")
    parts: list[np.ndarray] = []
    for index, start in enumerate(range(0, len(texts), shard_size)):
        shard = shard_dir / f"emb_{index:05d}.npy"
        if shard.exists():
            parts.append(np.load(shard))
            continue
        vectors = np.asarray(encode(texts[start : start + shard_size]), dtype=np.float32)
        tmp = shard.with_name(shard.name + ".tmp.npy")
        np.save(tmp, vectors)
        os.replace(tmp, shard)
        parts.append(vectors)
        print(f"embedded {min(start + shard_size, len(texts))}/{len(texts)}", file=sys.stderr)
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    emb = np.concatenate(parts)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)


def similarity_matrix(emb: np.ndarray) -> np.ndarray:
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    return sims


def edges_at_threshold(sims: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    rows, cols = np.nonzero(np.triu(sims >= threshold, k=1))
    return [(int(i), int(j), float(sims[i, j])) for i, j in zip(rows, cols)]


def sample_label_pairs(
    rows: list[dict[str, Any]],
    sims: np.ndarray,
    *,
    n_nn: int = 100,
    n_random: int = 100,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Nearest-neighbor pairs evenly spaced over similarity rank, plus uniform random pairs."""
    n = len(rows)
    if n < 2:
        raise ValueError("need at least two rows to sample pairs")
    nn: dict[tuple[int, int], float] = {}
    for i in range(n):
        j = int(np.argmax(sims[i]))
        nn[(min(i, j), max(i, j))] = float(sims[i, j])
    ranked = sorted(nn.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) <= n_nn:
        picked = ranked
    else:
        step = (len(ranked) - 1) / (n_nn - 1) if n_nn > 1 else 0
        picked = [ranked[round(k * step)] for k in range(n_nn)]
    chosen = {pair for pair, _ in picked}
    rng = random.Random(seed)
    random_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(random_pairs) < n_random and attempts < 100 * n_random:
        attempts += 1
        i, j = sorted(rng.sample(range(n), 2))
        if (i, j) in chosen:
            continue
        chosen.add((i, j))
        random_pairs.append((i, j))
    pairs = [(i, j, "nearest_neighbor") for (i, j), _ in picked]
    pairs += [(i, j, "random") for i, j in random_pairs]
    rng.shuffle(pairs)
    out = []
    for k, (i, j, source) in enumerate(pairs):
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        out.append(
            {
                "pair_id": f"pair-{k:03d}",
                "a": rows[a]["candidate_id"],
                "b": rows[b]["candidate_id"],
                "text_a": rows[a]["seed_text"],
                "text_b": rows[b]["seed_text"],
                "cosine": float(sims[i, j]),
                "source": source,
            }
        )
    return out


def choose_threshold(
    labeled: list[tuple[float, bool]],
    *,
    min_precision: float = 0.95,
) -> dict[str, Any]:
    """Lowest threshold (highest recall) among labeled similarities with precision >= target."""
    positives = sum(1 for _, dup in labeled if dup)
    if positives == 0:
        raise ValueError("no labeled duplicates; cannot calibrate a threshold")
    best: dict[str, Any] | None = None
    for t in sorted({sim for sim, _ in labeled}, reverse=True):
        predicted = [dup for sim, dup in labeled if sim >= t]
        tp = sum(predicted)
        precision = tp / len(predicted)
        if precision >= min_precision:
            best = {
                "threshold": t,
                "precision": precision,
                "recall": tp / positives,
                "predicted": len(predicted),
            }
    if best is None:
        raise ValueError(f"no threshold reaches precision {min_precision}")
    best.update({"labeled_pairs": len(labeled), "labeled_duplicates": positives, "min_precision": min_precision})
    return best


def _sentence_transformer_encoder(model_dir: Path, batch_size: int) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    return encode


def _load_and_embed(args: argparse.Namespace, encode: Encoder | None) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = load_accepted(args.quality, args.inventory)
    if encode is None:
        encode = _sentence_transformer_encoder(args.model_dir, args.batch_size)
    emb = embed_texts([normalize_text(row["seed_text"]) for row in rows], encode, shard_dir=args.out / "embeddings")
    return rows, emb


def _input_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quality_sha256": _sha256_file(args.quality),
        "inventory_sha256": _sha256_file(args.inventory),
        "model": json.loads(MODEL_MANIFEST_PATH.read_text()),
    }


def main(argv: list[str] | None = None, *, encode: Encoder | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--quality", type=Path, required=True, help="quality.jsonl from curation.quality")
        p.add_argument("--inventory", type=Path, required=True, help="inventory.jsonl with seed_text")
        p.add_argument("--out", type=Path, required=True, help="output directory (embedding shards are reused)")
        p.add_argument("--model-dir", type=Path, help="local snapshot of the pinned embedding model")
        p.add_argument("--batch-size", type=int, default=64)

    p_pairs = sub.add_parser("pairs", help="write blinded pairs for duplicate labeling")
    common(p_pairs)
    p_pairs.add_argument("--n-nn", type=int, default=100)
    p_pairs.add_argument("--n-random", type=int, default=100)
    p_pairs.add_argument("--seed", type=int, default=0)

    p_thr = sub.add_parser("threshold", help="freeze a threshold from labeled pairs")
    p_thr.add_argument("--key", type=Path, required=True, help="pairs_key.jsonl from the pairs step")
    p_thr.add_argument("--labels", type=Path, required=True, help='JSONL rows {"pair_id", "duplicate": bool}')
    p_thr.add_argument("--out", type=Path, required=True, help="threshold.json to write")
    p_thr.add_argument("--min-precision", type=float, default=0.95)

    p_cl = sub.add_parser("edges", help="write near-duplicate pairs at the frozen threshold")
    common(p_cl)
    p_cl.add_argument("--threshold-file", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "threshold":
        key = {row["pair_id"]: row for row in _read_jsonl(args.key)}
        labels = _read_jsonl(args.labels)
        missing = sorted(set(key) - {row["pair_id"] for row in labels})
        if missing:
            print(f"unlabeled pairs: {missing[:5]} ... ({len(missing)})", file=sys.stderr)
            return 1
        labeled = [(key[row["pair_id"]]["cosine"], bool(row["duplicate"])) for row in labels]
        result = choose_threshold(labeled, min_precision=args.min_precision)
        result["key_sha256"] = _sha256_file(args.key)
        result["labels_sha256"] = _sha256_file(args.labels)
        _atomic_write_text(args.out, json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), file=sys.stderr)
        return 0

    if encode is None and args.model_dir is None:
        print("--model-dir is required", file=sys.stderr)
        return 1
    rows, emb = _load_and_embed(args, encode)
    sims = similarity_matrix(emb)

    if args.command == "pairs":
        pairs = sample_label_pairs(rows, sims, n_nn=args.n_nn, n_random=args.n_random, seed=args.seed)
        _write_jsonl(args.out / "pairs_key.jsonl", pairs)
        blind = [{k: row[k] for k in ("pair_id", "text_a", "text_b")} for row in pairs]
        _write_jsonl(args.out / "pairs_blind.jsonl", blind)
        print(f"wrote {len(pairs)} pairs to {args.out}", file=sys.stderr)
        return 0

    threshold = json.loads(args.threshold_file.read_text())
    edges = edges_at_threshold(sims, float(threshold["threshold"]))
    roster = [{k: row[k] for k in ("candidate_id", "behavior", "scenario_realism", "eval_awareness")} for row in rows]
    _write_jsonl(args.out / "roster.jsonl", roster)
    _write_jsonl(
        args.out / "edges.jsonl",
        ({"a": rows[i]["candidate_id"], "b": rows[j]["candidate_id"], "cosine": s} for i, j, s in edges),
    )
    manifest = _input_manifest(args) | {
        "threshold": threshold,
        "threshold_file_sha256": _sha256_file(args.threshold_file),
        "accepted": len(rows),
        "edges": len(edges),
        "cross_behavior_edges": sum(1 for i, j, _ in edges if rows[i]["behavior"] != rows[j]["behavior"]),
        "candidates_with_neighbors": len({k for i, j, _ in edges for k in (i, j)}),
        "roster_sha256": _sha256_file(args.out / "roster.jsonl"),
        "edges_sha256": _sha256_file(args.out / "edges.jsonl"),
    }
    _atomic_write_text(args.out / "edges_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("accepted", "edges", "cross_behavior_edges")}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
