# Cogito — Training a Reasoning Model with GRPO

I trained a 1.5B language model to solve the **Countdown** numbers game using **GRPO** — the RL method behind DeepSeek-R1 — on a free Colab T4. No labelled reasoning examples, just reward functions.

Named after Descartes: *cogito ergo sum* — the model has to think to get rewarded.

---

## Results

| Metric (100 held-out Countdown problems) | Base model | After GRPO (250 steps, T4) |
|---|---|---|
| **Accuracy (hit the target)** | 0% | 30% |
| Format compliance | fails (LaTeX, `=` in expr, gives up) | 100% clean arithmetic expressions |
| Legal number usage | often uses numbers not in the set | consistently uses only given numbers |
| Reasoning style | rambles / hallucinates | short, focused search |

During training, the correctness reward spiked to **1.0 at step 135** — so the model can solve these problems, it just isn't reliable at single-shot inference yet. Format reward maxed by step 10; partial reward (writing valid expressions) improved ~4× by step 250. Getting to reliable accuracy needs around 1000 steps.

**Model on HF:** [harthik2006/cogito-countdown-grpo](https://huggingface.co/harthik2006/cogito-countdown-grpo)

---

## How GRPO works

For each problem, the model generates a group of 8 answers. A reward function scores each one — did it hit the target? did it use the right format? GRPO then pushes the model toward the above-average answers in each group, and away from the below-average ones. No labelled examples needed — the model figures out that reasoning earns more reward and starts doing it.

The reward signals I used:

| reward | fires when | value |
|---|---|---|
| correctness | `<answer>` hits the target | +2.0 |
| partial | legal expression using the given numbers (wrong value is fine) | +0.5 |
| strict format | full text is exactly `<reasoning>...</reasoning><answer>...</answer>` | +0.5 |
| soft format | both tags present in order | +0.25 |

The shaping rewards (partial + format) matter because correctness alone is too sparse early — the model needs something to learn from before it can solve any problems.

---

## Stack

| Piece | What |
|---|---|
| Compute | Google Colab free T4 (16 GB) |
| Training | Unsloth GRPO + vLLM fast rollouts |
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Task | Countdown numbers game |
| Tracking | Weights & Biases |
| Release | Hugging Face Hub |

---

## Files

```
cogito/
├── Cogito_GRPO.ipynb   # Colab notebook — full pipeline
├── task.py             # problem generator, prompt template, safe answer checker
├── reward.py           # 4 GRPO reward functions
├── evaluate.py         # accuracy scoring + before/after traces
├── app.py              # optional Gradio demo
├── test_task.py        # 10 tests   ┐
├── test_reward.py      # 15 tests   │  40 unit tests, no GPU needed
├── test_evaluate.py    #  9 tests   │
└── test_app.py         #  6 tests   ┘
```

---

## Running locally (no GPU)

```bash
cd cogito
python test_task.py && python test_reward.py && python test_evaluate.py && python test_app.py
```

All 40 tests pass without a GPU or any model downloaded.

## Training on Colab

1. Upload `Cogito_GRPO.ipynb` to [colab.research.google.com](https://colab.research.google.com) → Runtime → T4 GPU
2. Run Phase 0 cells (install + load model, ~10 min)
3. Upload `task.py`, `reward.py`, `evaluate.py` to the Files panel
4. Run top to bottom: baseline eval → GRPO training → trained eval → push to HF

Add Colab secrets: `WANDB_API_KEY` and `HF_TOKEN` (write scope). Edit `HF_USERNAME` in the push cell.

---

## Credits

Built on [Unsloth](https://github.com/unslothai/unsloth)'s GRPO recipe, [Qwen2.5](https://huggingface.co/Qwen), and TRL's GRPOTrainer. Countdown-as-RL-task inspired by TinyZero and the DeepSeek-R1 reproductions.
