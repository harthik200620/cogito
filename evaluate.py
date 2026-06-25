"""Evaluate base vs trained model accuracy on held-out Countdown problems.

generate_one and run_eval are GPU-only (they call vLLM). Everything else —
score_records, save_traces, save_before_after, summarize — is pure Python and
unit-testable without any GPU or heavy deps.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import task


def make_eval_dataset(n: int = 100, seed: int = 2025, exclude: Optional[set] = None):
    """Build n held-out problems, disjoint from training if exclude is given."""
    return task.build_dataset(n=n, seed=seed, exclude=exclude)


def generate_one(
    model,
    tokenizer,
    messages: list[dict],
    lora_request=None,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Generate one completion for a chat messages list. Requires GPU + vLLM.

    Pass lora_request=model.load_lora("grpo_saved_lora") for the trained model,
    or leave it None for the base model.
    """
    from vllm import SamplingParams

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
    )
    out = model.fast_generate(text, sampling_params=sampling_params, lora_request=lora_request)
    return out[0].outputs[0].text


def run_eval(model, tokenizer, dataset, lora_request=None, **gen_kwargs) -> dict:
    """Run one model over the full eval dataset and return accuracy + records. Requires GPU."""
    records = []
    for row in dataset:
        nums, target = row["nums"], row["target"]
        messages = row["prompt"]
        completion = generate_one(model, tokenizer, messages, lora_request=lora_request, **gen_kwargs)
        answer = task.extract_answer(completion)
        correct = task.is_correct(answer, nums, target)
        records.append(
            {
                "nums": list(nums),
                "target": target,
                "completion": completion,
                "answer": answer,
                "correct": bool(correct),
            }
        )
    stats = score_records(records)
    return {"accuracy": stats["accuracy"], "n": stats["n"], "n_correct": stats["n_correct"], "records": records}


def score_records(records: list[dict]) -> dict:
    """Aggregate records into {accuracy, n_correct, n}. Pure, no GPU needed."""
    n = len(records)
    n_correct = sum(1 for r in records if r.get("correct"))
    accuracy = (n_correct / n) if n else 0.0
    return {"accuracy": accuracy, "n_correct": n_correct, "n": n}


def _snippet(completion: Optional[str], max_chars: int = 300) -> str:
    """Pull a short reasoning snippet for the Markdown report."""
    import re

    if not isinstance(completion, str) or not completion.strip():
        return "(no output)"
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", completion, re.DOTALL | re.IGNORECASE)
    text = m.group(1).strip() if m else completion.strip()
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


def save_traces(records: list[dict], path: str) -> str:
    """Write eval records to path as JSON. Returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return path


def _mark(correct: bool) -> str:
    return "[CORRECT]" if correct else "[wrong]"


def save_before_after(base_records: list[dict], trained_records: list[dict], path_md: str) -> str:
    """Write a side-by-side before/after Markdown report. Returns the path."""
    parent = os.path.dirname(os.path.abspath(path_md))
    if parent:
        os.makedirs(parent, exist_ok=True)

    base_stats = score_records(base_records)
    trained_stats = score_records(trained_records)

    lines: list[str] = []
    lines.append("# Countdown GRPO -- Before vs After")
    lines.append("")
    lines.append(summarize({"accuracy": base_stats["accuracy"], "n": base_stats["n"]},
                           {"accuracy": trained_stats["accuracy"], "n": trained_stats["n"]}))
    lines.append("")
    lines.append(
        "Each block is the SAME problem solved by the base model and by the "
        "GRPO-trained model. " + _mark(True) + " means the answer expression hit "
        "the target."
    )
    lines.append("")

    n = min(len(base_records), len(trained_records))
    for i in range(n):
        b = base_records[i]
        t = trained_records[i]
        b_key = task.problem_key({"nums": b["nums"], "target": b["target"]})
        t_key = task.problem_key({"nums": t["nums"], "target": t["target"]})
        match_note = "" if b_key == t_key else "  (WARNING: problem mismatch)"

        lines.append(f"## Problem {i + 1}{match_note}")
        lines.append(f"- Numbers: `{b['nums']}`")
        lines.append(f"- Target: `{b['target']}`")
        lines.append("")
        lines.append(f"**BASE** {_mark(bool(b.get('correct')))} answer: `{b.get('answer')}`")
        lines.append("")
        lines.append(f"> {_snippet(b.get('completion'))}")
        lines.append("")
        lines.append(f"**TRAINED** {_mark(bool(t.get('correct')))} answer: `{t.get('answer')}`")
        lines.append("")
        lines.append(f"> {_snippet(t.get('completion'))}")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path_md


def summarize(base_eval: dict, trained_eval: dict) -> str:
    """One-line headline: 'Base X% -> Trained Y% (+Z pts) on N held-out problems'."""
    base_acc = base_eval.get("accuracy", 0.0)
    trained_acc = trained_eval.get("accuracy", 0.0)
    delta = trained_acc - base_acc
    n = trained_eval.get("n", base_eval.get("n", 0))
    return (
        f"Base {base_acc * 100:.1f}% -> Trained {trained_acc * 100:.1f}% "
        f"({delta * 100:+.1f} pts) on {n} held-out problems"
    )


def main():
    eval_ds = make_eval_dataset()
    print(f"Built held-out eval set: {len(eval_ds)} problems (columns: {eval_ds.column_names}).")
    print("Need a loaded model + tokenizer for run_eval.")
    return eval_ds


if __name__ == "__main__":
    main()
