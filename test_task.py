"""Tests for task.py — run with `python cogito/test_task.py` or `pytest`.

No GPU needed. Validates: problems are actually solvable, the answer checker is
correct, illegal/unsafe expressions are rejected, and dataset building works.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `import task` from anywhere

import task


# --- a brute-force Countdown solver, used only to PROVE generated problems
#     are solvable (independent of how task.py builds them) -------------------
def _solvable(nums, target):
    def helper(values):
        if any(abs(v - target) < 1e-9 for v in values):
            return True
        n = len(values)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = values[i], values[j]
                rest = [values[k] for k in range(n) if k != i and k != j]
                cands = [a + b, a - b, a * b]
                if abs(b) > 1e-9:
                    cands.append(a / b)
                for c in cands:
                    if helper(rest + [c]):
                        return True
        return False

    return helper([float(x) for x in nums])


def test_generated_problems_are_solvable():
    rng = random.Random(123)
    for _ in range(60):
        p = task.generate_problem(n_numbers=4, max_number=50, rng=rng)
        assert len(p["nums"]) == 4
        assert 10 <= p["target"] <= 1000
        assert _solvable(p["nums"], p["target"]), f"unsolvable: {p}"


def test_make_prompt_shape():
    msgs = task.make_prompt([1, 2, 3, 4], 24)
    assert msgs[0]["role"] == "system" and "Countdown" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "Target: 24" in msgs[1]["content"] and "1, 2, 3, 4" in msgs[1]["content"]


def test_extract_answer():
    assert task.extract_answer("<reasoning>x</reasoning><answer> 1+2 </answer>") == "1+2"
    # takes the LAST answer block if several
    assert task.extract_answer("<answer>1</answer> junk <answer>2</answer>") == "2"
    assert task.extract_answer("no tags here") is None
    assert task.extract_answer(None) is None


def test_evaluate_correct():
    r = task.evaluate_expression("(9 * 50) + (3 * 7)", [3, 7, 9, 50], 471)
    assert r["correct"] and r["legal"] and abs(r["value"] - 471) < 1e-9


def test_evaluate_wrong_value():
    r = task.evaluate_expression("3 + 7", [3, 7, 9, 50], 471)
    assert r["legal"] and not r["correct"] and r["reason"] == "wrong value"


def test_evaluate_illegal_numbers():
    # 100 is not among the given numbers
    r = task.evaluate_expression("100 + 371", [3, 7, 9, 50], 471)
    assert not r["legal"] and not r["correct"]


def test_evaluate_reuse_number_too_often():
    # only one 3 is available, expression uses it twice
    r = task.evaluate_expression("3 + 3", [3, 7], 6)
    assert not r["legal"]


def test_evaluate_rejects_unsafe_or_unsupported():
    for expr in ["__import__('os').system('echo hi')", "2 ** 10", "pow(2,3)", "a + b", ""]:
        r = task.evaluate_expression(expr, [2, 3, 10], 8)
        assert not r["valid"] or not r["correct"], f"should not be correct: {expr!r}"
    # specifically, code-injection style input must be flagged invalid
    assert task.evaluate_expression("__import__('os')", [1], 1)["valid"] is False


def test_is_correct():
    assert task.is_correct("2 * 3 * 4", [2, 3, 4, 5], 24)
    assert not task.is_correct("2 * 3", [2, 3, 4, 5], 24)


def test_build_dataset_unique_and_disjoint():
    train = task.build_dataset(n=15, seed=1)
    assert len(train) == 15
    assert set(train.column_names) == {"prompt", "nums", "target"}
    train_keys = {task.problem_key({"nums": r["nums"], "target": r["target"]}) for r in train}
    assert len(train_keys) == 15  # all unique
    # eval excludes train -> disjoint
    eval_ds = task.build_dataset(n=15, seed=2, exclude=train_keys)
    eval_keys = {task.problem_key({"nums": r["nums"], "target": r["target"]}) for r in eval_ds}
    assert train_keys.isdisjoint(eval_keys)


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {_name}: {type(e).__name__}: {e}")
    print("\n" + ("ALL PASSED" if failures == 0 else f"{failures} FAILED"))
    sys.exit(1 if failures else 0)
