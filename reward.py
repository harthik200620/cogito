"""4 GRPO reward functions for the Countdown task. Pure Python, no torch needed.

TRL calls each function with (prompts, completions, **dataset_columns) and expects
a list[float] back — one score per completion. The four signals together build a
reward ladder: format first, valid expressions next, correct answer last.
"""

from __future__ import annotations

import re
from typing import Any

import task


def _text(completion: Any) -> str:
    """Normalize a TRL completion to a plain string — handles both conversational and str formats."""
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


_STRICT_RE = re.compile(r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$", re.DOTALL)
_SOFT_RE = re.compile(r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>", re.DOTALL)


def correctness_reward_func(prompts, completions, target, nums, **kwargs) -> list[float]:
    """2.0 if the <answer> actually solves the puzzle, 0.0 otherwise."""
    texts = [_text(c) for c in completions]
    return [
        2.0 if task.is_correct(task.extract_answer(t), n, tgt) else 0.0
        for t, tgt, n in zip(texts, target, nums)
    ]


def partial_reward_func(prompts, completions, target, nums, **kwargs) -> list[float]:
    """0.5 if the answer is a legal parseable expression using the given numbers (even if wrong value)."""
    texts = [_text(c) for c in completions]
    out: list[float] = []
    for t, tgt, n in zip(texts, target, nums):
        info = task.evaluate_expression(task.extract_answer(t), n, tgt)
        out.append(0.5 if (info["valid"] and info["legal"]) else 0.0)
    return out


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """0.5 if the full text is exactly <reasoning>...</reasoning><answer>...</answer>."""
    texts = [_text(c) for c in completions]
    return [0.5 if _STRICT_RE.match(t) else 0.0 for t in texts]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """0.25 if both tags appear in the right order (extra text around them is fine)."""
    texts = [_text(c) for c in completions]
    return [0.25 if _SOFT_RE.search(t) else 0.0 for t in texts]


# TRL sums these per completion. Order: main objective first, shaping rewards after.
REWARD_FUNCS = [
    correctness_reward_func,
    partial_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
]
