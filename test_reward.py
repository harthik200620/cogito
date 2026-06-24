"""test_reward.py — local unit tests for reward.py (Countdown GRPO rewards).

Runs two ways:
  - ``python test_reward.py``  -> prints PASS/FAIL per test, exits non-zero on failure.
  - ``pytest test_reward.py``  -> standard test discovery (test_* functions).

ASCII-only console output (Windows cp1252 — no emoji / unicode in prints).
"""

import os
import sys

# Make `import task` / `import reward` work regardless of the current directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reward  # noqa: E402
import task  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: build completions in BOTH TRL shapes from a body of text.
# ---------------------------------------------------------------------------


def _conv(text):
    """Conversational completion: list with one assistant message."""
    return [{"role": "assistant", "content": text}]


def _str(text):
    """Plain-string completion."""
    return text


def _wrap(reasoning, answer):
    """Build a correctly-formatted completion body."""
    return "<reasoning>\n{}\n</reasoning>\n<answer>\n{}\n</answer>".format(reasoning, answer)


# A solvable instance from the spec: (9 * 50) + (3 * 7) == 471.
NUMS = [3, 7, 9, 50]
TARGET = 471
GOOD_EXPR = "(9 * 50) + (3 * 7)"          # correct
WRONG_LEGAL_EXPR = "(9 * 50) + (3 - 7)"   # legal numbers, value 446 != 471
ILLEGAL_EXPR = "(8 * 50) + (3 * 7)"       # uses 8, which is not in NUMS

GOOD_FULL = _wrap("try 9*50=450, plus 3*7=21", GOOD_EXPR)
WRONG_FULL = _wrap("try 9*50=450, plus 3-7=-4", WRONG_LEGAL_EXPR)
ILLEGAL_FULL = _wrap("use 8 and 50", ILLEGAL_EXPR)


# ---------------------------------------------------------------------------
# Tests. Each returns None and raises AssertionError on failure.
# ---------------------------------------------------------------------------


def test_correctness_true_both_shapes():
    """correctness == 2.0 for the right expression, in conv AND string form."""
    for make in (_conv, _str):
        r = reward.correctness_reward_func(
            prompts=[None], completions=[make(GOOD_FULL)], target=[TARGET], nums=[NUMS]
        )
        assert r == [2.0], "expected [2.0], got {} ({})".format(r, make.__name__)


def test_correctness_false_for_wrong():
    """correctness == 0.0 when the expression does not hit the target."""
    r = reward.correctness_reward_func(
        prompts=[None], completions=[_conv(WRONG_FULL)], target=[TARGET], nums=[NUMS]
    )
    assert r == [0.0], "expected [0.0] for wrong expr, got {}".format(r)


def test_partial_legal_but_wrong():
    """partial == 0.5 for a legal-but-wrong expression (shaping reward)."""
    for make in (_conv, _str):
        r = reward.partial_reward_func(
            prompts=[None], completions=[make(WRONG_FULL)], target=[TARGET], nums=[NUMS]
        )
        assert r == [0.5], "expected [0.5], got {} ({})".format(r, make.__name__)


def test_partial_zero_for_illegal():
    """partial == 0.0 when the expression uses numbers not in the given set."""
    r = reward.partial_reward_func(
        prompts=[None], completions=[_conv(ILLEGAL_FULL)], target=[TARGET], nums=[NUMS]
    )
    assert r == [0.0], "expected [0.0] for illegal numbers, got {}".format(r)


def test_partial_correct_is_also_legal():
    """A correct expression is by definition legal, so partial == 0.5 too."""
    r = reward.partial_reward_func(
        prompts=[None], completions=[_str(GOOD_FULL)], target=[TARGET], nums=[NUMS]
    )
    assert r == [0.5], "expected [0.5] for correct (legal) expr, got {}".format(r)


def test_strict_fires_on_exact_structure():
    """strict == 0.5 for exact <reasoning>..</reasoning><answer>..</answer>."""
    for make in (_conv, _str):
        r = reward.strict_format_reward_func(completions=[make(GOOD_FULL)])
        assert r == [0.5], "expected [0.5], got {} ({})".format(r, make.__name__)


def test_strict_zero_when_extra_text():
    """strict == 0.0 when there is leading prose before <reasoning>."""
    messy = "Sure! here goes:\n" + GOOD_FULL
    r = reward.strict_format_reward_func(completions=[_str(messy)])
    assert r == [0.0], "expected [0.0] for messy text, got {}".format(r)


def test_strict_zero_when_only_answer():
    """strict == 0.0 when the <reasoning> block is missing entirely."""
    r = reward.strict_format_reward_func(completions=[_str("<answer>\n5\n</answer>")])
    assert r == [0.0], "expected [0.0] when reasoning missing, got {}".format(r)


def test_soft_fires_on_loose_structure():
    """soft == 0.25 when both tags appear in order, even with extra text."""
    messy = "Let me think.\n" + GOOD_FULL + "\nDone!"
    for make in (_conv, _str):
        r = reward.soft_format_reward_func(completions=[make(messy)])
        assert r == [0.25], "expected [0.25], got {} ({})".format(r, make.__name__)


def test_soft_zero_when_out_of_order():
    """soft == 0.0 when <answer> precedes <reasoning> (wrong order)."""
    bad = "<answer>\n5\n</answer>\n<reasoning>\noops\n</reasoning>"
    r = reward.soft_format_reward_func(completions=[_str(bad)])
    assert r == [0.0], "expected [0.0] for out-of-order tags, got {}".format(r)


def test_batch_alignment():
    """A batch of 3 (mixed shapes) stays aligned, per-sample, across rewards."""
    completions = [_conv(GOOD_FULL), _str(WRONG_FULL), _conv(ILLEGAL_FULL)]
    targets = [TARGET, TARGET, TARGET]
    nums = [NUMS, NUMS, NUMS]

    corr = reward.correctness_reward_func(
        prompts=[None] * 3, completions=completions, target=targets, nums=nums
    )
    part = reward.partial_reward_func(
        prompts=[None] * 3, completions=completions, target=targets, nums=nums
    )
    assert corr == [2.0, 0.0, 0.0], "correctness misaligned: {}".format(corr)
    # good=legal, wrong=legal, illegal=not legal
    assert part == [0.5, 0.5, 0.0], "partial misaligned: {}".format(part)
    assert len(corr) == 3 and len(part) == 3


def test_missing_tags_no_crash():
    """No <answer> at all -> all rewards 0.0 and nothing raises."""
    completions = [_conv("just some text, no tags here"), _str("")]
    targets = [TARGET, TARGET]
    nums = [NUMS, NUMS]
    corr = reward.correctness_reward_func(
        prompts=[None, None], completions=completions, target=targets, nums=nums
    )
    part = reward.partial_reward_func(
        prompts=[None, None], completions=completions, target=targets, nums=nums
    )
    strict = reward.strict_format_reward_func(completions=completions)
    soft = reward.soft_format_reward_func(completions=completions)
    assert corr == [0.0, 0.0], corr
    assert part == [0.0, 0.0], part
    assert strict == [0.0, 0.0], strict
    assert soft == [0.0, 0.0], soft


def test_malformed_completion_shapes_no_crash():
    """Odd shapes (empty list, dict without content, None) degrade to 0.0."""
    weird = [[], [{"role": "assistant"}], None, 12345]
    targets = [TARGET] * len(weird)
    nums = [NUMS] * len(weird)
    corr = reward.correctness_reward_func(
        prompts=[None] * len(weird), completions=weird, target=targets, nums=nums
    )
    soft = reward.soft_format_reward_func(completions=weird)
    assert corr == [0.0, 0.0, 0.0, 0.0], corr
    assert soft == [0.0, 0.0, 0.0, 0.0], soft


def test_reward_funcs_exported():
    """REWARD_FUNCS lists all four functions in the documented order."""
    assert reward.REWARD_FUNCS == [
        reward.correctness_reward_func,
        reward.partial_reward_func,
        reward.strict_format_reward_func,
        reward.soft_format_reward_func,
    ]


def test_extra_kwargs_ignored():
    """Extra dataset columns passed by TRL must not break the call."""
    r = reward.correctness_reward_func(
        prompts=[None],
        completions=[_conv(GOOD_FULL)],
        target=[TARGET],
        nums=[NUMS],
        some_extra_column=["whatever"],
        prompt_ids=[[1, 2, 3]],
    )
    assert r == [2.0], r


# ---------------------------------------------------------------------------
# Standalone runner (pytest-compatible: it just calls the test_* functions).
# ---------------------------------------------------------------------------


def _all_tests():
    g = globals()
    return [(name, g[name]) for name in sorted(g) if name.startswith("test_")]


def main():
    failures = 0
    for name, fn in _all_tests():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL {}: {}".format(name, exc))
        else:
            print("PASS {}".format(name))
    total = len(_all_tests())
    print("-" * 40)
    print("{}/{} tests passed".format(total - failures, total))
    if failures:
        print("RESULT: FAILURE ({} failed)".format(failures))
        return 1
    print("RESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
