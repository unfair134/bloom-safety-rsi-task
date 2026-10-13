#!/usr/bin/env python3
"""XSTest over-refusal baseline, using the official inspect_evals grading.

  VLLM_BASE_URL=... python -m judge.xstest --csv xstest_prompts.csv --out run_xstest --subset safe --target defender

The prompts come from the authors' release (github.com/paul-rottger/exaggerated-safety,
`xstest_prompts.csv`, 250 safe + 200 unsafe), because the Hugging Face copy that
inspect_evals loads is gated. Grading template, instructions, system message and
refusal metric are imported from inspect_evals 0.23.0.

Deviation from the official config (temperature 0, max_tokens 256): the target
uses the same sampling and thinking setup as the Bloom target, because 256
tokens cannot fit Qwen3.5's thinking. The grader is Claude Opus 4.7.
"""
import argparse
import csv
import os

from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message
from inspect_evals.xstest.xstest import refusal_rate, scorer_instructions, scorer_template

CLAUDE = "anthropic/claude-opus-4-7"


def dataset(path, subset):
    rows = [r for r in csv.DictReader(open(path)) if r["label"] == subset]
    return MemoryDataset([Sample(input=r["prompt"], id=f"xstest_{r['id']}",
                                 metadata={k: r[k] for k in ("type", "label", "focus", "note")})
                          for r in rows], name=f"xstest_{subset}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subset", nargs="+", default=["safe", "unsafe"])
    p.add_argument("--target", default="qwen3.5-9b")
    p.add_argument("--effort", default="medium")
    p.add_argument("--connections", type=int, default=32)
    args = p.parse_args()

    grader = get_model(CLAUDE, config=GenerateConfig(reasoning_effort=args.effort, max_connections=20))
    target = get_model(f"vllm/{args.target}", base_url=os.environ["VLLM_BASE_URL"],
                       api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                       config=GenerateConfig(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                                             max_tokens=16384, max_connections=args.connections))
    tasks = [Task(name=f"xstest_{s}", dataset=dataset(args.csv, s),
                  solver=[system_message("You are a helpful assistant."), generate()],
                  scorer=model_graded_qa(model=grader, template=scorer_template,
                                         instructions=scorer_instructions),
                  metrics=[refusal_rate()])
             for s in args.subset]
    eval_set(tasks, log_dir=args.out, model=target, max_tasks=len(tasks), retry_attempts=3)


if __name__ == "__main__":
    main()
