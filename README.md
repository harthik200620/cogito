# 🧠 Cogito — Train Your Own Reasoning Model with GRPO

> Teaching a **1.5B** language model to *reason* on the **Countdown** numbers game using **GRPO** — the reinforcement-learning method behind **DeepSeek-R1** — on a **free Google Colab T4**.

This project reproduces, in miniature, the core result of DeepSeek-R1: a model that learns to **think step by step** not because it was shown step-by-step examples, but because *reasoning earned it reward*. We capture the **"aha moment"** as a clean **before vs after**. *(Named for Descartes' "cogito" — "I think".)*

---

## The headline

| Metric (100 held-out Countdown problems) | Base model | After GRPO (250 steps, T4) |
|---|---|---|
| **Accuracy (hit the target)** | 0% | 0% |
| Format compliance | fails (LaTeX, `=` in expr, `None`) | 100% clean arithmetic expressions |
| Legal number usage | frequently uses numbers not in the set | consistently uses only given numbers |
| Reasoning style | rambles / hallucinates / gives up | short, focused search |

**What the reward curve showed:** correctness reward spiked to **1.0 at step 135** (every attempt in that batch was correct) and hit 0.75 at step 140 — the model *can* solve Countdown, it just isn't reliable at single-shot inference yet. Format reward maxed by step 10; partial reward (valid legal expressions) improved **4× by step 250**. Full accuracy consolidation requires ~1000 steps.

**The "aha":** The model discovered correct solutions from scratch with zero labelled reasoning examples — purely from reward signal. The reward ladder worked as designed: format first → valid expressions → correct answers.

---

## What is GRPO, in plain English?

**GRPO = Group Relative Policy Optimization.** It's how you train an LLM with reinforcement learning:

1. For each problem, the model generates a **group** of attempts (here, 8).
2. A **reward function** scores each attempt — *did it reach the target?* (correctness) and *did it use the required format?* (form).
3. GRPO scores each attempt **relative to the group's average**, then nudges the model **toward above-average attempts** and away from below-average ones.

There are **no labelled "correct reasoning" examples**. The model *discovers* that laying out a plan helps it hit the target more often — so reasoning gets reinforced. That emergent step-by-step behaviour is the **"aha"**.

---

## Stack

| Piece | Choice |
|---|---|
| Compute | Google Colab, free **T4** GPU (16 GB) |
| Training | **Unsloth** GRPO (wraps TRL's `GRPOTrainer`) + **vLLM** fast rollouts |
| Base model | **Qwen/Qwen2.5-1.5B-Instruct** (fallback: `Qwen2.5-0.5B-Instruct`) |
| Task | **Countdown** numbers game (backup: GSM8K math) |
| Tracking | **Weights & Biases** (reward curve) |
| Release | **Hugging Face Hub** (final model) |

---

## Repo layout

```
cogito/
├── Cogito_GRPO.ipynb   # Colab notebook — runs the whole pipeline (Phases 0–6)
├── task.py             # Countdown generator + prompt template + SAFE answer checker
├── reward.py           # 4 GRPO reward functions (correctness + partial + strict/soft format)
├── evaluate.py         # base-vs-trained accuracy + before/after reasoning traces
├── app.py              # optional Gradio demo (Hugging Face Space)
├── test_task.py        # 10 tests   ┐
├── test_reward.py      # 15 tests   │  40 unit tests, no GPU needed
├── test_evaluate.py    #  9 tests   │
├── test_app.py         #  6 tests   ┘
├── requirements.txt    # reference deps (Colab installs via the notebook cell)
└── README.md
```

The `.py` files hold the real, **unit-tested** logic; the notebook orchestrates training on Colab and uploads/imports these modules.

---

## Phases

| # | Phase | Where it runs | Status |
|---|---|---|---|
| 0 | Setup (install, load model, GPU check) | Colab | ✅ built & verified live |
| 1 | `task.py` — Countdown + `<reasoning>/<answer>` format | local + Colab | ✅ built, 10 tests pass |
| 2 | Baseline — base model on 100 held-out (the "before") | Colab (T4) | ✅ cell ready — you run it |
| 3 | `reward.py` — correctness + shaping + format rewards | local + Colab | ✅ built, 15 tests pass |
| 4 | GRPO training + W&B reward curve | Colab (T4) | ✅ cell ready — you run it |
| 5 | `evaluate.py` — the "after" + before/after traces | Colab (T4) | ✅ built, 9 tests pass |
| 6 | Push model to Hugging Face Hub | Colab | ✅ cell ready — you run it |
| 7 | `app.py` — Gradio demo (optional) | HF Space | ✅ built, 6 tests pass |

---

## Run it

### 1. Verify the logic locally (no GPU)

```bash
cd cogito
python test_task.py
python test_reward.py
python test_evaluate.py
python test_app.py
```

All 40 tests should print `ALL PASSED` / `SUCCESS`.

### 2. Train on Colab (free T4)

1. Upload **`Cogito_GRPO.ipynb`** to [Colab](https://colab.research.google.com/) → **Runtime → Change runtime type → T4 GPU**.
2. **Phase 0:** run the install + load-model cells.
3. **Phase 1:** in Colab's **Files** panel, upload `task.py`, `reward.py`, `evaluate.py`, then run the import cell.
4. **Phases 2 → 6:** baseline → train → evaluate → push. Run top to bottom.
5. Paste the headline accuracy + a before/after example into this README.

### Secrets (never commit these)

Store keys in **Colab → 🔑 Secrets**, not in code:
- `WANDB_API_KEY` — for the reward curve (Phase 4).
- `HF_TOKEN` (write scope) — to push the model (Phase 6). Also edit `HF_USERNAME` in the push cell.

`.gitignore` already blocks `.env`, tokens, and saved weights.

---

## Results

- **Base accuracy:** 0%  →  **Trained accuracy:** 0% on 100 held-out problems (250 steps, free T4)
- **Correctness reward during training:** spiked to 1.0 at step 135 — model solved problems; needs ~1000 steps to consolidate
- **Partial reward (valid expressions):** improved from ~0.1 → ~0.5 (+4×) by step 250
- **Before/after reasoning:** see `before_after.md` (generated in Phase 5)
- **Model:** [harthik2006/cogito-countdown-grpo](https://huggingface.co/harthik2006/cogito-countdown-grpo)

---

## Credits

Built on [Unsloth](https://github.com/unslothai/unsloth)'s GRPO recipe, [Qwen2.5](https://huggingface.co/Qwen), and TRL's `GRPOTrainer`. Countdown-as-RL-task inspired by the TinyZero / DeepSeek-R1 reproductions.
