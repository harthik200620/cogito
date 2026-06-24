"""app.py - Gradio demo for the "Cogito" Countdown GRPO model.

The user enters a list of numbers and a target; the trained model reasons inside
<reasoning>...</reasoning> and gives an arithmetic expression inside
<answer>...</answer>. We surface the reasoning, the extracted answer, and a
verdict on whether the expression legally hits the target.

The pure helpers (parse_numbers, format_verdict) and the answer-checking logic
live in task.py and are unit-tested in test_app.py WITHOUT torch / gradio /
unsloth, so all heavy imports are deferred into functions.

--------------------------------------------------------------------------------
Hugging Face Space setup
--------------------------------------------------------------------------------
1. Set MODEL_ID below to the model you pushed in Phase 6
   (e.g. "your-username/cogito-countdown-grpo").
2. requirements.txt should contain:

       gradio
       unsloth

   or, if you are not using Unsloth in the Space:

       gradio
       transformers
       peft
       accelerate
       torch

3. A GPU Space is recommended (the model runs in 4-bit; CPU will be very slow).
4. Set app.py as the Space's entry point (the default) and launch.
"""

from __future__ import annotations

import task

# --- Phase 6 model -----------------------------------------------------------
MODEL_ID = "YOUR_HF_USERNAME/cogito-countdown-grpo"  # TODO: set to your pushed model

# Cache for the lazily-loaded (model, tokenizer) pair.
_MODEL = None
_TOKENIZER = None


# ---------------------------------------------------------------------------
# Pure helpers (no model / gradio / torch) -- these are what test_app.py tests
# ---------------------------------------------------------------------------
def parse_numbers(s: str) -> list[int]:
    """Parse "3, 7, 9, 50" or "3 7 9 50" (any mix of commas/spaces) -> [int]."""
    if not s:
        return []
    tokens = s.replace(",", " ").split()
    nums = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        nums.append(int(tok))
    return nums


def format_verdict(expr: str, nums: list[int], target: int) -> str:
    """Human-readable verdict for a candidate expression, via task.evaluate_expression.

    Examples:
      "Correct! 9*50 + 3*7 = 471"
      "Not quite: that equals 470, target was 471"
      "Invalid: uses numbers not in the list"
      "Invalid: not a valid arithmetic expression"
      "Invalid: no expression found"
    """
    if not expr:
        return "Invalid: no expression found"

    result = task.evaluate_expression(expr, nums, target)
    expr_clean = expr.strip()

    if not result["valid"]:
        return "Invalid: not a valid arithmetic expression"

    value = result["value"]
    # Show whole numbers without a trailing ".0".
    if isinstance(value, float) and value.is_integer():
        value_str = str(int(value))
    else:
        value_str = str(value)

    if result["correct"]:
        return "Correct! {} = {}".format(expr_clean, value_str)
    if not result["legal"]:
        return "Invalid: uses numbers not in the list"
    return "Not quite: that equals {}, target was {}".format(value_str, target)


def split_reasoning(text: str) -> str:
    """Pull the contents of the LAST <reasoning>...</reasoning> block, else the
    raw text (stripped). Pure string helper."""
    import re

    if not isinstance(text, str):
        return ""
    matches = re.findall(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Lazy model loading -- all heavy imports happen INSIDE this function
# ---------------------------------------------------------------------------
def get_model():
    """Load the model + tokenizer once and cache them.

    Tries Unsloth's FastLanguageModel first (4-bit, fast inference); falls back
    to plain transformers if Unsloth is not installed. Heavy imports (torch,
    unsloth, transformers) are deferred here so importing this module for tests
    never requires them.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_ID,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    _MODEL, _TOKENIZER = model, tokenizer
    return _MODEL, _TOKENIZER


# ---------------------------------------------------------------------------
# Inference + verdict (drives the UI)
# ---------------------------------------------------------------------------
def solve(numbers_str: str, target):
    """Run the full pipeline for one problem.

    Returns (reasoning_text, answer_text, verdict_text). Heavy imports are
    deferred into get_model(); this function is only called from the UI.
    """
    try:
        nums = parse_numbers(numbers_str)
    except ValueError:
        return (
            "",
            "",
            "Invalid: could not parse the numbers (use e.g. 3, 7, 9, 50).",
        )

    if not nums:
        return ("", "", "Invalid: please enter at least one number.")

    try:
        target = int(target)
    except (TypeError, ValueError):
        return ("", "", "Invalid: please enter a numeric target.")

    model, tokenizer = get_model()

    messages = task.make_prompt(nums, target)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )
    # Decode only the newly generated continuation.
    generated = output_ids[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    reasoning_text = split_reasoning(text)
    answer_text = task.extract_answer(text) or ""
    verdict_text = format_verdict(answer_text, nums, target)

    return reasoning_text, answer_text, verdict_text


# ---------------------------------------------------------------------------
# Gradio UI (gradio imported lazily so the module imports without it)
# ---------------------------------------------------------------------------
def build_demo():
    import gradio as gr

    with gr.Blocks(title="Cogito - Countdown reasoner") as demo:
        gr.Markdown(
            "# Cogito - Countdown reasoner\n"
            "Enter some numbers and a target. The GRPO-trained model reasons "
            "step by step, then proposes an arithmetic expression. "
            "Each number may be used at most once."
        )
        with gr.Row():
            numbers_in = gr.Textbox(
                label="Numbers",
                placeholder="Numbers, e.g. 3, 7, 9, 50",
            )
            target_in = gr.Number(label="Target", precision=0)
        solve_btn = gr.Button("Solve", variant="primary")

        reasoning_out = gr.Textbox(label="Reasoning", lines=10)
        answer_out = gr.Textbox(label="Answer")
        verdict_out = gr.Textbox(label="Hit the target?")

        solve_btn.click(
            fn=solve,
            inputs=[numbers_in, target_in],
            outputs=[reasoning_out, answer_out, verdict_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
