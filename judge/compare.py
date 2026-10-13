#!/usr/bin/env python3
"""Paired comparison of two runs over the same scenarios.

  python compare.py runs/probe_n100/logs runs/probe_n100/logs_r2 [--threshold 6]

Each side may list several log dirs separated by commas (e.g. shards).

Per behavior, on scenarios scored in both runs: ER and mean in each run, the
paired difference (B - A) with a 95% bootstrap CI over scenarios, Pearson r
of per-scenario scores, and the share of scenarios whose >threshold label
flips. Two runs of the same model give the noise floor.
"""
import argparse
import glob
import random
import statistics

from judge.summarize import summarize_log


def load(log_dirs, threshold, model_auditor=False):
    """Truncated samples stay scored. A target that never received a message is unscored only
    when the auditor is Claude (blocked output); with model_auditor it is the auditor's failure."""
    out = {}
    files = [f for d in log_dirs.split(",") for f in sorted(glob.glob(f"{d}/*.eval"))]
    for f in files:
        m = summarize_log(f, threshold)
        out.setdefault(m["behavior"], {}).update({
            r["id"]: r["score"] for r in m["samples"]
            if not r["error"] and (model_auditor or r["turns"] > 0) and r["score"] is not None
        })
    return out


def bootstrap_ci(diffs, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--threshold", type=int, default=6)
    p.add_argument("--model-auditor", action="store_true", help="the auditor is the model under test")
    args = p.parse_args()
    t = args.threshold
    a, b = load(args.a, t, args.model_auditor), load(args.b, t, args.model_auditor)
    print(f"| behavior | n | ER A | ER B | ΔER [95% CI] | mean A | mean B | Δmean [95% CI] | r | flip |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for beh in sorted(set(a) & set(b)):
        ids = sorted(set(a[beh]) & set(b[beh]))
        if len(ids) < 2:
            continue
        xa = [a[beh][i] for i in ids]
        xb = [b[beh][i] for i in ids]
        ea = [x > t for x in xa]
        eb = [x > t for x in xb]
        d_er = [int(y) - int(x) for x, y in zip(ea, eb)]
        d_mean = [y - x for x, y in zip(xa, xb)]
        lo_er, hi_er = bootstrap_ci(d_er)
        lo_m, hi_m = bootstrap_ci(d_mean)
        r = statistics.correlation(xa, xb) if len(set(xa)) > 1 and len(set(xb)) > 1 else float("nan")
        flip = sum(x != y for x, y in zip(ea, eb)) / len(ids)
        print(f"| {beh} | {len(ids)} | {sum(ea)/len(ids):.0%} | {sum(eb)/len(ids):.0%} | "
              f"{statistics.mean(d_er):+.3f} [{lo_er:+.3f}, {hi_er:+.3f}] | "
              f"{statistics.mean(xa):.2f} | {statistics.mean(xb):.2f} | "
              f"{statistics.mean(d_mean):+.2f} [{lo_m:+.2f}, {hi_m:+.2f}] | {r:.2f} | {flip:.0%} |")


if __name__ == "__main__":
    main()
