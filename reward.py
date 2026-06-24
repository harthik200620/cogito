"""reward.py — GRPO reward functions for the "Cogito" Countdown project.

These are the scalar reward signals GRPO uses to push Qwen2.5-1.5B toward
*reasoning then answering* on the Countdown numbers game. They are PURE PYTHON
(no torch / trl import) so they unit-test anywhere, but their signatures match
exactly what TRL's GRPOTrainer calls at train time.

TRL reward contract
-------------------
Each reward function has the signature::

    def f(prompts, completions, **kwargs) -> list[float]

and returns ONE float per completion in the batch. TRL forwards the dataset
columns as KEYWORD ARGUMENTS that are LISTS aligned to the batch — here that is
``target: list[int]`` and ``nums: list[list[int]]`` (see task.build_dataset,
whose columns are prompt / nums / target). We always accept ``**kwargs`` so any
extra columns TRL passes never break the call.

`completions` can arrive in two shapes depending on the model / config:
  - CONVERSATIONAL: each item is ``[{"role": "assistant", "content": "..."}]``
  - PLAIN STRING:   each item is just ``"..."``
``_text()`` normalizes both to a raw string.

The RL intuition for the bundle: `correctness` is the real objective (did it
solve the puzzle), but it is sparse — early in training almost every sample
scores 0, giving GRPO nothing to climb. The other three are *shaping* rewards:
`partial` rewards a legal, well-formed expression even when the value is wrong,
and the two format rewards reward producing the required
<reasoning>/<answer> structure at all. Together they create a gradient from
"emits gibberish" -> "emits the right shape" -> "emits a legal expression" ->
"emits the correct answer", so the model always has a nearby reward to chase.
"""

from __future__ import annotations

import re
from typing import Any

import task  # task.py lives alongside this file; pure-Python, already tested.

# ---------------------------------------------------------------------------
# Completion normalizer
# ---------------------------------------------------------------------------


def _text(completion: Any) -> str:
    """Return the raw assistant text from one completion, in either TRL shape.

    - Conversational: ``[{"role": "assistant", "content": "..."}]`` -> the
      "content" of the (last) message.
    - Plain string: returned unchanged.
    Anything unexpected degrades to "" so a reward func never raises mid-batch
    (a single bad sample must not crash training).
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            return content if isinstance(content, str) else ""
        if isinstance(last, str):
            return last
    return ""


# ---------------------------------------------------------------------------
# Format regexes
# ---------------------------------------------------------------------------
# STRICT: the WHOLE text must be exactly one <reasoning> block followed by one
# <answer> block (trailing whitespace allowed). This is the format task.py asks
# the model to produce verbatim, so matching it rewards clean obedience.
_STRICT_RE = re.compile(r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$", re.DOTALL)

# SOFT: both blocks merely appear, in order, somewhere in the text (extra prose
# before/after/between is tolerated). A looser, easier-to-earn shaping signal.
_SOFT_RE = re.compile(r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>", re.DOTALL)


# ---------------------------------------------------------------------------
# 1. Correctness — the main objective
# ---------------------------------------------------------------------------


def correctness_reward_func(prompts, completions, target, nums, **kwargs) -> list[float]:
    """2.0 if the extracted <answer> actually solves the puzzle, else 0.0.

    This is the true task signal: it fires only when the model's final
    expression legally combines the given numbers to equal the target. It is
    deliberately the largest reward so that, once the model can produce legal
    expressions, GRPO is pulled hardest toward *correct* ones. Sparse on its own
    — hence the shaping rewards below.
    """
    texts = [_text(c) for c in completions]
    return [
        2.0 if task.is_correct(task.extract_answer(t), n, tgt) else 0.0
        for t, tgt, n in zip(texts, target, nums)
    ]


# ---------------------------------------------------------------------------
# 2. Partial — reward legal, well-formed attempts (shaping)
# ---------------------------------------------------------------------------


def partial_reward_func(prompts, completions, target, nums, **kwargs) -> list[float]:
    """0.5 if the answer is a LEGAL, parseable expression over the given numbers.

    Rewards getting the *mechanics* right — a syntactically valid expression
    (parses, only + - * / and parentheses) that uses only the provided numbers,
    each no more than allowed — even when the value is wrong. This densifies the
    reward landscape: the model learns to form valid Countdown expressions long
    before it can hit exact targets, giving GRPO a foothold while correctness is
    still mostly zero.
    """
    texts = [_text(c) for c in completions]
    out: list[float] = []
    for t, tgt, n in zip(texts, target, nums):
        info = task.evaluate_expression(task.extract_answer(t), n, tgt)
        out.append(0.5 if (info["valid"] and info["legal"]) else 0.0)
    return out


# ---------------------------------------------------------------------------
# 3. Strict format — exact output structure (shaping)
# ---------------------------------------------------------------------------


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """0.5 if the FULL text is exactly <reasoning>...</reasoning><answer>...</answer>.

    Rewards producing the precise format task.SYSTEM_PROMPT demands and nothing
    else. Clean structure makes the answer reliably extractable and discourages
    the model from rambling outside the tags.
    """
    texts = [_text(c) for c in completions]
    return [0.5 if _STRICT_RE.match(t) else 0.0 for t in texts]


# ---------------------------------------------------------------------------
# 4. Soft format — loose structure (shaping)
# ---------------------------------------------------------------------------


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """0.25 if a <reasoning> block and an <answer> block both appear, in order.

    A gentler version of the format reward: it pays out as soon as the model
    emits both tags in the right order, even with stray text around them. This
    is usually the *first* reward a fresh model can earn, so it bootstraps the
    structure that the strict/partial/correctness rewards then refine.
    """
    texts = [_text(c) for c in completions]
    return [0.25 if _SOFT_RE.search(t) else 0.0 for t in texts]


# Order: main objective first, then shaping rewards. TRL sums these per sample.
REWARD_FUNCS = [
    correctness_reward_func,
    partial_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
]
