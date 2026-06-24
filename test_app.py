import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""test_app.py - tests for the PURE logic in app.py.

Covers parse_numbers and format_verdict only. These import nothing heavy:
importing app.py must NOT require torch / gradio / unsloth (all deferred into
functions), so this file passes locally even though none of those are installed.

ASCII-only output (Windows cp1252 console). Run directly:
    python test_app.py
Exits non-zero on any failure. Also pytest-compatible.
"""

import app


def test_parse_numbers_comma():
    assert app.parse_numbers("3, 7, 9, 50") == [3, 7, 9, 50]


def test_parse_numbers_space():
    assert app.parse_numbers("3 7 9 50") == [3, 7, 9, 50]


def test_parse_numbers_mixed_and_messy():
    # Extra/irregular whitespace and a trailing comma still parse cleanly.
    assert app.parse_numbers("  3 ,7,  9   50 ") == [3, 7, 9, 50]


def test_format_verdict_correct():
    nums = [3, 7, 9, 50]
    verdict = app.format_verdict("(9 * 50) + (3 * 7)", nums, 471)
    assert verdict.startswith("Correct!"), verdict
    assert "471" in verdict, verdict


def test_format_verdict_not_quite():
    # (9 * 50) + (3 * 6) = 468, but we ask for target 471 -> wrong value.
    # Use only legal numbers so it is "not quite", not "invalid".
    nums = [3, 6, 9, 50]
    verdict = app.format_verdict("(9 * 50) + (3 * 6)", nums, 471)
    assert verdict.startswith("Not quite"), verdict
    assert "468" in verdict, verdict
    assert "471" in verdict, verdict


def test_format_verdict_invalid_illegal_numbers():
    # 100 is not in the list -> illegal numbers.
    nums = [3, 7, 9, 50]
    verdict = app.format_verdict("100 + 1", nums, 101)
    assert verdict.startswith("Invalid"), verdict
    assert "not in the list" in verdict, verdict


def _run():
    tests = [
        ("parse_numbers comma form", test_parse_numbers_comma),
        ("parse_numbers space form", test_parse_numbers_space),
        ("parse_numbers mixed/messy", test_parse_numbers_mixed_and_messy),
        ("format_verdict correct", test_format_verdict_correct),
        ("format_verdict not quite", test_format_verdict_not_quite),
        ("format_verdict invalid (illegal numbers)", test_format_verdict_invalid_illegal_numbers),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS:", name)
        except AssertionError as e:
            failures += 1
            print("FAIL:", name, "->", e)
        except Exception as e:
            failures += 1
            print("FAIL:", name, "-> unexpected", type(e).__name__, e)

    print("-" * 40)
    if failures:
        print("FAILED {} of {} tests".format(failures, len(tests)))
    else:
        print("ALL {} TESTS PASSED".format(len(tests)))
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
