"""Countdown game: generate problems, prompt template, and answer checker.

The game: combine numbers with + - * / (each used at most once) to hit a target.
Pure Python, no GPU needed — imported by the notebook, reward.py, and evaluate.py.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from collections import Counter
from typing import Optional


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
    """Return the chat messages for one problem."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(nums=list(nums), target=target)},
    ]


_OPS = ["+", "-", "*", "/"]


def _random_expr(values: list[int], rng: random.Random):
    """Fold a list of ints into a value using random ops. Returns (value, expr) or (None, None)."""
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
        else:
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
    """Generate one solvable Countdown problem -> {"nums": [...], "target": int}.

    Builds a random expression over random numbers and uses the result as the target,
    so a solution is always guaranteed to exist.
    """
    rng = rng or random
    for _ in range(max_tries):
        nums = [rng.randint(1, max_number) for _ in range(n_numbers)]
        order = nums[:]
        rng.shuffle(order)
        value, _expr = _random_expr(order, rng)
        if value is None:
            continue
        if abs(value - round(value)) < 1e-9:
            t = int(round(value))
            if min_target <= t <= max_target:
                return {"nums": nums, "target": t}
    # Fallback: use sum of numbers as target (always solvable)
    for _ in range(max_tries):
        nums = [rng.randint(1, max_number) for _ in range(n_numbers)]
        s = sum(nums)
        if min_target <= s <= max_target:
            return {"nums": nums, "target": s}
    return {"nums": nums, "target": sum(nums)}


def problem_key(problem: dict):
    """Hashable identity of a problem, for deduplication."""
    return (tuple(sorted(problem["nums"])), problem["target"])


def build_dataset(
    n: int,
    n_numbers: int = 4,
    max_number: int = 50,
    seed: int = 0,
    exclude: Optional[set] = None,
):
    """Build a HuggingFace Dataset of n unique Countdown problems.

    Columns: prompt (chat messages), nums, target.
    Pass exclude=train_keys to keep eval disjoint from training.
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


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> Optional[str]:
    """Return the last <answer>...</answer> block contents, else None."""
    if not isinstance(text, str):
        return None
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


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
    return [int(tok) for tok in re.findall(r"\d+", expr)]


def evaluate_expression(expr: Optional[str], nums: list[int], target: int) -> dict:
    """Check a candidate answer expression.

    Returns dict with: valid, value, legal, correct, reason.
    Uses a safe AST walker instead of eval() so model output can never run arbitrary code.
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
    """True if expr legally combines the given numbers to equal target."""
    return evaluate_expression(expr, nums, target)["correct"]


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(3):
        p = generate_problem(rng=rng)
        msgs = make_prompt(p["nums"], p["target"])
        print(msgs[1]["content"], "\n")
