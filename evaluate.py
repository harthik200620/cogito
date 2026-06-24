"""evaluate.py — base-vs-trained accuracy on held-out Countdown problems.

This is the "did GRPO actually teach it to reason?" measurement. We:
  1. build a held-out eval set of Countdown problems, DISJOINT from training,
  2. generate one completion per problem from the BASE model and again from the
     GRPO-trained model (LoRA loaded),
  3. score each completion with task.is_correct (pure, deterministic),
  4. emit two artifacts: raw traces (JSON) and a readable before/after Markdown
     pairing the SAME problems side by side — the "aha moment" report.

DESIGN: the only GPU-dependent code is `generate_one` (and `run_eval`, which
calls it). vllm is imported LAZILY inside `generate_one`, so EVERYTHING else
(dataset building, scoring, artifact writing, the headline summary) imports and
unit-tests on a plain CPU machine with no torch / vllm installed.

Used by:
  - the notebook : after training, call run_eval twice and save_before_after once.
  - test_evaluate.py : exercises the pure scoring + artifact paths (no GPU).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import task

# ---------------------------------------------------------------------------
# 1. Held-out eval set (pure: just wraps task.build_dataset)
# ---------------------------------------------------------------------------


def make_eval_dataset(n: int = 100, seed: int = 2025, exclude: Optional[set] = None):
    """Build n held-out Countdown problems for evaluation.

    `exclude` is a set of task.problem_key()s (typically every key in the TRAIN
    set) so the eval problems are guaranteed DISJOINT from training — otherwise
    "accuracy" would be partly memorisation. A different default `seed` from the
    train set is a second line of defence. Returns a HuggingFace Dataset with
    columns: prompt, nums, target.
    """
    return task.build_dataset(n=n, seed=seed, exclude=exclude)


# ---------------------------------------------------------------------------
# 2. Generation — THE ONLY GPU-DEPENDENT PART (vllm imported lazily)
# ---------------------------------------------------------------------------


def generate_one(
    model,
    tokenizer,
    messages: list[dict],
    lora_request=None,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Generate one completion for a chat `messages` list. GPU REQUIRED.

    Mirrors the generation pattern already used in the notebook: render the chat
    template to a single string, then call the Unsloth/vLLM `fast_generate`.
    Pass `lora_request = model.load_lora("grpo_saved_lora")` to evaluate the
    TRAINED adapter; leave it None for the BASE model.

    vllm is imported HERE (lazily) so the rest of this module stays importable
    and testable on machines without vllm/torch installed.
    """
    from vllm import SamplingParams  # lazy: only needed when actually generating

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
    )
    out = model.fast_generate(text, sampling_params=sampling_params, lora_request=lora_request)
    return out[0].outputs[0].text


# ---------------------------------------------------------------------------
# 3. Run an evaluation pass (GPU: calls generate_one per row)
# ---------------------------------------------------------------------------


def run_eval(model, tokenizer, dataset, lora_request=None, **gen_kwargs) -> dict:
    """Generate + score one model over the whole eval `dataset`. GPU REQUIRED.

    For every row we build a record:
        {nums, target, completion, answer, correct}
    where `answer` = task.extract_answer(completion) and
    `correct` = task.is_correct(answer, nums, target).

    Returns {"accuracy", "n", "n_correct", "records"}. The heavy lifting of
    turning records into stats lives in `score_records` so it can be unit-tested
    without a model.
    """
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


# ---------------------------------------------------------------------------
# 4. Pure scoring + artifacts (NO GPU — fully unit-testable)
# ---------------------------------------------------------------------------


def score_records(records: list[dict]) -> dict:
    """Aggregate records -> {accuracy, n_correct, n}. PURE / deterministic.

    accuracy is in [0, 1]; an empty list scores 0.0 (avoids divide-by-zero).
    """
    n = len(records)
    n_correct = sum(1 for r in records if r.get("correct"))
    accuracy = (n_correct / n) if n else 0.0
    return {"accuracy": accuracy, "n_correct": n_correct, "n": n}


def _snippet(completion: Optional[str], max_chars: int = 300) -> str:
    """A short, single-line-ish reasoning snippet for the Markdown report.

    Prefer the <reasoning>...</reasoning> body if present (that's the part we
    actually care about reading); otherwise fall back to the raw completion.
    Truncated to keep the before/after table readable.
    """
    import re

    if not isinstance(completion, str) or not completion.strip():
        return "(no output)"
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", completion, re.DOTALL | re.IGNORECASE)
    text = m.group(1).strip() if m else completion.strip()
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


def save_traces(records: list[dict], path: str) -> str:
    """Write the full eval records to `path` as pretty JSON. Returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return path


def _mark(correct: bool) -> str:
    """ASCII-only correctness marker (Windows cp1252 safe — no emoji)."""
    return "[CORRECT]" if correct else "[wrong]"


def save_before_after(base_records: list[dict], trained_records: list[dict], path_md: str) -> str:
    """Write a readable before/after Markdown report pairing the SAME problems.

    Records are paired by position; we additionally assert the problem matches
    (same problem_key) so a row never compares two different problems. For each
    problem we show: Numbers, Target, the BASE answer + correctness + a reasoning
    snippet, and the TRAINED answer + correctness + a reasoning snippet. This is
    the headline "aha moment" artifact. Returns the path.
    """
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
        # Pair safety: same underlying problem on both sides.
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
    """One-line headline, e.g.
    'Base 6.0% -> Trained 51.0% (+45.0 pts) on 100 held-out problems'.

    Accepts either a full run_eval dict or any dict with 'accuracy' (and
    optionally 'n'). ASCII-only.
    """
    base_acc = base_eval.get("accuracy", 0.0)
    trained_acc = trained_eval.get("accuracy", 0.0)
    delta = trained_acc - base_acc
    n = trained_eval.get("n", base_eval.get("n", 0))
    return (
        f"Base {base_acc * 100:.1f}% -> Trained {trained_acc * 100:.1f}% "
        f"({delta * 100:+.1f} pts) on {n} held-out problems"
    )


# ---------------------------------------------------------------------------
# 5. Notebook entry point
# ---------------------------------------------------------------------------


def main():
    """Build the held-out eval set and explain the (GPU) steps to run it.

    Importing/running this WITHOUT a model is fine — it just prepares the eval
    set and prints the recipe. `run_eval` is the part that needs a loaded
    Unsloth FastLanguageModel + tokenizer with vLLM fast inference.

    Typical notebook usage (after training):

        import evaluate
        # keep eval disjoint from the problems the model trained on:
        train_keys = {task.problem_key({"nums": r["nums"], "target": r["target"]})
                      for r in train_dataset}
        eval_ds = evaluate.make_eval_dataset(n=100, seed=2025, exclude=train_keys)

        base = evaluate.run_eval(model, tokenizer, eval_ds, lora_request=None)
        lora = model.load_lora("grpo_saved_lora")
        trained = evaluate.run_eval(model, tokenizer, eval_ds, lora_request=lora)

        print(evaluate.summarize(base, trained))
        evaluate.save_traces(base["records"], "eval_base_traces.json")
        evaluate.save_traces(trained["records"], "eval_trained_traces.json")
        evaluate.save_before_after(base["records"], trained["records"],
                                   "before_after.md")
    """
    eval_ds = make_eval_dataset()
    print(f"Built held-out eval set: {len(eval_ds)} problems "
          f"(columns: {eval_ds.column_names}).")
    print("A loaded model + tokenizer (Unsloth FastLanguageModel, vLLM) is "
          "required for run_eval — see this function's docstring for the recipe.")
    return eval_ds


if __name__ == "__main__":
    main()
