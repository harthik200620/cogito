"""task.py — Countdown problem generation, prompt template, and answer checking.

The Countdown numbers game: given a small set of numbers and a target, find an
arithmetic expression (using + - * / and parentheses, each number used at most
once) that equals the target. e.g. nums=[3, 7, 9, 50], target=471 -> 9*50 + 3*7.

This module is PURE PYTHON (no torch / GPU) so it can be unit-tested anywhere and
imported both locally and inside the Colab notebook.

Used by:
  - reward.py    : the correctness reward calls is_correct() / evaluate_expression()
  - evaluate.py  : builds held-out problems and checks the model's answers
  - the notebook : builds the train / eval datasets via build_dataset()
"""

from __future__ import annotations

import ast
import operator
import random
import re
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Prompt + required output format
# ---------------------------------------------------------------------------
# The model must wrap its thinking in <reasoning>...</reasoning> and its final
# arithmetic expression in <answer>...</answer>. reward.py grades BOTH:
#   - format reward      : is that structure present?
#   - correctness reward : does the <answer> expression actually hit the target?
# Forcing a fixed shape is what makes "did it reason?" and "did it solve it?"
# mechanically checkable — the core trick behind GRPO on a reasoning task.

SYSTEM_PROMPT = (
    "You are solving the Countdown numbers game. You are given a list of numbers "
    "and a target number. Combine the numbers using + - * / and parentheses to "
    "reach the target. You may use each number at most once, and you do not have "
    "to use all of them.\n"
    "First think step by step inside <reasoning> </reasoning>: try combinations "
    "and check what they equal. Then put ONLY the final arithmetic expression "
    "inside <answer> </answer>.\n"
    "Respond in exactly this format:\n"
    "<reasoning>\n...your step-by-step search...\n</reasoning>\n"
    "<answer>\nexpression that equals the target\n</answer>"
)

USER_TEMPLATE = "Numbers: {nums}\nTarget: {target}"


def make_prompt(nums: list[int], target: int) -> list[dict]:
    """Return chat messages for one problem (this is the GRPO 'prompt' column)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(nums=list(nums), target=target)},
    ]


# ---------------------------------------------------------------------------
# 2. Problem generation (guaranteed solvable)
# ---------------------------------------------------------------------------
_OPS = ["+", "-", "*", "/"]


def _random_expr(values: list[int], rng: random.Random):
    """Fold a list of ints into one value using random ops.

    Returns (value: float, expr: str) or (None, None) if a division would be
    non-integer (we keep intermediate divisions clean, classic-Countdown style).
    """
    items = [(float(v), str(v)) for v in values]
    while len(items) > 1:
        (av, astr) = items.pop(rng.randrange(len(items)))
        (bv, bstr) = items.pop(rng.randrange(len(items)))
        op = rng.choice(_OPS)
        if op == "+":
            cv = av + bv
        elif op == "-":
            cv = av - bv
        elif op == "*":
            cv = av * bv
        else:  # division must be exact and non-zero
            if bv == 0 or abs(av % bv) > 1e-9:
                return None, None
            cv = av / bv
        items.append((cv, f"({astr} {op} {bstr})"))
    return items[0][0], items[0][1]


def generate_problem(
    n_numbers: int = 4,
    max_number: int = 50,
    min_target: int = 10,
    max_target: int = 1000,
    rng: Optional[random.Random] = None,
    max_tries: int = 200,
) -> dict:
    """Generate ONE solvable Countdown problem -> {"nums": [...], "target": int}.

    Strategy: pick random numbers, build a random expression over them, and use
    its value as the target. Because the target is *built from* the numbers, a
    solution is guaranteed to exist. We keep only positive-integer targets inside
    [min_target, max_target].
    """
    rng = rng or random
    for _ in range(max_tries):
        nums = [rng.randint(1, max_number) for _ in range(n_numbers)]
        order = nums[:]
        rng.shuffle(order)
        value, _expr = _random_expr(order, rng)
        if value is None:
            continue
        if abs(value - round(value)) < 1e-9:  # clean integer?
            t = int(round(value))
            if min_target <= t <= max_target:
                return {"nums": nums, "target": t}
    # Fallback (always solvable): a plain sum, retried until it lands in range.
    for _ in range(max_tries):
        nums = [rng.randint(1, max_number) for _ in range(n_numbers)]
        s = sum(nums)
        if min_target <= s <= max_target:
            return {"nums": nums, "target": s}
    return {"nums": nums, "target": sum(nums)}  # last resort: still solvable


def problem_key(problem: dict):
    """Hashable identity of a problem, for de-duplicating / disjoint splits."""
    return (tuple(sorted(problem["nums"])), problem["target"])


def build_dataset(
    n: int,
    n_numbers: int = 4,
    max_number: int = 50,
    seed: int = 0,
    exclude: Optional[set] = None,
):
    """Build a HuggingFace Dataset of n UNIQUE Countdown problems for GRPO.

    Columns: prompt (chat messages), nums (list[int]), target (int).
    `exclude` is a set of problem_key()s to skip (use it to keep eval disjoint
    from train). Imports `datasets` lazily so task.py stays importable without it.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    seen = set(exclude) if exclude else set()
    rows = []
    while len(rows) < n:
        p = generate_problem(n_numbers=n_numbers, max_number=max_number, rng=rng)
        key = problem_key(p)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"prompt": make_prompt(p["nums"], p["target"]), "nums": p["nums"], "target": p["target"]})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# 3. Answer extraction + safe checking
# ---------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> Optional[str]:
    """Return the contents of the LAST <answer>...</answer> block, else None."""
    if not isinstance(text, str):
        return None
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


# Only these AST node types are allowed — this is how we evaluate the model's
# expression WITHOUT running arbitrary code (never use eval() on model output).
_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
                   ast.Mult: operator.mul, ast.Div: operator.truediv}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    raise ValueError("disallowed expression")


def _numbers_in(expr: str) -> list[int]:
    """Integer literals appearing in the expression."""
    return [int(tok) for tok in re.findall(r"\d+", expr)]


def evaluate_expression(expr: Optional[str], nums: list[int], target: int) -> dict:
    """Check a candidate Countdown answer expression.

    Returns:
      valid   - parses and uses only + - * / ( ) and integers
      value   - the numeric result (or None if invalid)
      legal   - uses only the given numbers, each at most as many times as given
      correct - legal AND value == target
      reason  - short human-readable explanation
    """
    bad = {"valid": False, "value": None, "legal": False, "correct": False}
    if not expr or not isinstance(expr, str):
        return {**bad, "reason": "empty"}
    expr = expr.strip()
    try:
        value = _safe_eval(ast.parse(expr, mode="eval"))
    except Exception:
        return {**bad, "reason": "parse/eval error"}
    used, given = Counter(_numbers_in(expr)), Counter(nums)
    legal = len(used) >= 1 and all(used[k] <= given.get(k, 0) for k in used)
    correct = legal and abs(value - target) < 1e-9
    reason = "correct" if correct else ("wrong value" if legal else "uses illegal numbers")
    return {"valid": True, "value": value, "legal": legal, "correct": correct, "reason": reason}


def is_correct(expr: Optional[str], nums: list[int], target: int) -> bool:
    """True iff `expr` legally combines the given numbers to equal `target`."""
    return evaluate_expression(expr, nums, target)["correct"]


if __name__ == "__main__":
    # Tiny demo so `python task.py` shows something useful.
    rng = random.Random(0)
    for _ in range(3):
        p = generate_problem(rng=rng)
        msgs = make_prompt(p["nums"], p["target"])
        print(msgs[1]["content"], "\n")
