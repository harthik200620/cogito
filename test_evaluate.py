"""Tests for evaluate.py — run with `python cogito/test_evaluate.py` or `pytest`.

NO GPU. We test only the pure / non-model parts of evaluate.py:
  - score_records accuracy math,
  - make_eval_dataset row count + disjointness from a train-key set,
  - save_traces + save_before_after write NON-EMPTY files (temp dir, cleaned up),
  - summarize produces the expected headline string.

The GPU-dependent generate_one / run_eval are NOT exercised here (vllm is imported
lazily inside generate_one, so importing evaluate.py needs no torch/vllm). ASCII
output only (Windows cp1252 — no emoji/unicode).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow `import evaluate`/`import task`

import json
import shutil
import tempfile

import evaluate
import task


def test_score_records_accuracy_math():
    recs = [
        {"correct": True},
        {"correct": False},
        {"correct": True},
        {"correct": False},
    ]
    s = evaluate.score_records(recs)
    assert s["n"] == 4
    assert s["n_correct"] == 2
    assert abs(s["accuracy"] - 0.5) < 1e-9


def test_score_records_empty_is_zero():
    s = evaluate.score_records([])
    assert s["n"] == 0 and s["n_correct"] == 0 and s["accuracy"] == 0.0


def test_score_records_all_correct():
    s = evaluate.score_records([{"correct": True}, {"correct": True}])
    assert s["n_correct"] == 2 and abs(s["accuracy"] - 1.0) < 1e-9


def test_make_eval_dataset_count_and_columns():
    ds = evaluate.make_eval_dataset(n=12, seed=2025)
    assert len(ds) == 12
    assert set(ds.column_names) == {"prompt", "nums", "target"}


def test_make_eval_dataset_disjoint_from_train():
    # a stand-in "train" set, then eval must avoid every train problem_key
    train = task.build_dataset(n=20, seed=1)
    train_keys = {
        task.problem_key({"nums": r["nums"], "target": r["target"]}) for r in train
    }
    eval_ds = evaluate.make_eval_dataset(n=20, seed=2025, exclude=train_keys)
    eval_keys = {
        task.problem_key({"nums": r["nums"], "target": r["target"]}) for r in eval_ds
    }
    assert len(eval_ds) == 20
    assert train_keys.isdisjoint(eval_keys)


def _sample_records():
    """Two paired (same-problem) record lists: base mostly wrong, trained right."""
    base = [
        {"nums": [3, 7, 9, 50], "target": 471,
         "completion": "<reasoning>I guess.</reasoning><answer>3+7</answer>",
         "answer": "3+7", "correct": False},
        {"nums": [2, 3, 4, 5], "target": 24,
         "completion": "<reasoning>hmm</reasoning><answer>2+3</answer>",
         "answer": "2+3", "correct": False},
    ]
    trained = [
        {"nums": [3, 7, 9, 50], "target": 471,
         "completion": "<reasoning>9*50=450, 3*7=21, 450+21=471.</reasoning><answer>(9*50)+(3*7)</answer>",
         "answer": "(9*50)+(3*7)", "correct": True},
        {"nums": [2, 3, 4, 5], "target": 24,
         "completion": "<reasoning>2*3*4=24.</reasoning><answer>2*3*4</answer>",
         "answer": "2*3*4", "correct": True},
    ]
    return base, trained


def test_save_traces_writes_nonempty_file():
    base, _ = _sample_records()
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "traces.json")
        out = evaluate.save_traces(base, path)
        assert out == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        # round-trips back to the same records
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == len(base)
        assert loaded[0]["answer"] == base[0]["answer"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_before_after_writes_nonempty_readable_file():
    base, trained = _sample_records()
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "before_after.md")
        out = evaluate.save_before_after(base, trained, path)
        assert out == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        with open(path, "r", encoding="utf-8") as f:
            md = f.read()
        # headline + both problems present + ASCII correctness markers
        assert "Base 0.0% -> Trained 100.0%" in md
        assert "Problem 1" in md and "Problem 2" in md
        assert "BASE" in md and "TRAINED" in md
        assert "[CORRECT]" in md and "[wrong]" in md
        # must be pure ASCII (Windows cp1252 safety)
        md.encode("ascii")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_summarize_headline_string():
    base = {"accuracy": 0.06, "n": 100}
    trained = {"accuracy": 0.51, "n": 100}
    s = evaluate.summarize(base, trained)
    assert s == "Base 6.0% -> Trained 51.0% (+45.0 pts) on 100 held-out problems"
    s.encode("ascii")  # ASCII-only


def test_summarize_accepts_run_eval_dicts():
    # summarize should also work on full run_eval outputs (extra keys ignored)
    base = {"accuracy": 0.0, "n": 50, "n_correct": 0, "records": []}
    trained = {"accuracy": 0.4, "n": 50, "n_correct": 20, "records": []}
    s = evaluate.summarize(base, trained)
    assert s == "Base 0.0% -> Trained 40.0% (+40.0 pts) on 50 held-out problems"


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
