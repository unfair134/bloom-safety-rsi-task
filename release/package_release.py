"""Assemble the Hugging Face dataset folder from an export and the reference files.

    python package_release.py --release release_v2 --reference reference_v2 --card ../release/README.md \
        --key-file heldout_key.env --out hf_dataset

The held-out outlines are tarred and encrypted with
    openssl enc -aes-256-cbc -pbkdf2 -iter 600000 -salt -pass env:HELDOUT_KEY
and only the ciphertext is copied to the output. Decrypt with
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 -pass env:HELDOUT_KEY -in heldout/heldout.tar.enc | tar -x
The data card's <FILL_*> placeholders are replaced with the plaintext and ciphertext SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

HELDOUT = ("final_test.jsonl", "feedback_validation.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(release: Path) -> None:
    files = json.loads((release / "manifest.json").read_text())["files"]
    bad = [rel for rel, h in files.items() if sha256(release / rel) != h]
    if bad:
        raise SystemExit(f"manifest mismatch: {bad}")


def read_key(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("HELDOUT_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no HELDOUT_KEY in {path}")


def encrypt(src_dir: Path, dst: Path, key: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "heldout.tar"
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tar:
            for name in HELDOUT:
                info = tar.gettarinfo(src_dir / name, arcname=f"heldout/{name}")
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with open(src_dir / name, "rb") as f:
                    tar.addfile(info, f)
        env = {**os.environ, "HELDOUT_KEY": key}
        base = ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "600000", "-pass", "env:HELDOUT_KEY"]
        subprocess.run(base + ["-salt", "-in", str(tar_path), "-out", str(dst)], env=env, check=True)
        roundtrip = subprocess.run(base + ["-d", "-in", str(dst)], env=env, check=True, capture_output=True).stdout
        if roundtrip != tar_path.read_bytes():
            raise SystemExit("decryption round trip failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--card", type=Path, required=True)
    ap.add_argument("--key-file", type=Path, required=True)
    ap.add_argument("--extra", action="append", default=[], help="SRC=DEST: copy a file (e.g. a license) to DEST in the output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    check_manifest(args.release)
    shutil.copytree(args.release / "public", args.out / "public")
    shutil.copytree(args.reference, args.out / "reference")
    for spec in args.extra:
        src, dest = spec.split("=", 1)
        shutil.copyfile(src, args.out / dest)
    (args.out / "heldout").mkdir()
    enc = args.out / "heldout" / "heldout.tar.enc"
    encrypt(args.release / "heldout", enc, read_key(args.key_file))
    fills = {f"<FILL_{name}>": sha256(args.release / "heldout" / name) for name in HELDOUT}
    fills["<FILL_heldout.tar.enc>"] = sha256(enc)
    card = args.card.read_text()
    for k, v in fills.items():
        if k not in card:
            raise SystemExit(f"{k} not in data card")
        card = card.replace(k, v)
    (args.out / "README.md").write_text(card)
    files = sorted(p for p in args.out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "dataset_version": json.loads((args.release / "manifest.json").read_text())["dataset_version"],
        "heldout_plaintext_sha256": {name: fills[f"<FILL_{name}>"] for name in HELDOUT},
        "files": {str(p.relative_to(args.out)): sha256(p) for p in files},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files) + 1, **fills}, indent=1))


if __name__ == "__main__":
    main()
