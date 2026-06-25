"""Gradio demo for the Cogito Countdown model.

Set MODEL_ID to your pushed HF repo, then run `python app.py` or deploy as a
Gradio Space. A GPU Space is recommended (model runs in 4-bit; CPU is very slow).
Heavy imports (torch, unsloth, gradio) are deferred so test_app.py works without them.
"""

from __future__ import annotations

import task

MODEL_ID = "harthik2006/cogito-countdown-grpo"

_MODEL = None
_TOKENIZER = None


def parse_numbers(s: str) -> list[int]:
    """Parse "3, 7, 9, 50" or "3 7 9 50" into a list of ints."""
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
    """Return a human-readable verdict for a candidate expression."""
    if not expr:
        return "Invalid: no expression found"

    result = task.evaluate_expression(expr, nums, target)
    expr_clean = expr.strip()

    if not result["valid"]:
        return "Invalid: not a valid arithmetic expression"

    value = result["value"]
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
    """Pull the last <reasoning>...</reasoning> block, or the full text if no tags."""
    import re

    if not isinstance(text, str):
        return ""
    matches = re.findall(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def get_model():
    """Load model + tokenizer once and cache. Tries Unsloth first, falls back to transformers."""
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


def solve(numbers_str: str, target):
    """Run the full pipeline for one problem. Returns (reasoning, answer, verdict)."""
    try:
        nums = parse_numbers(numbers_str)
    except ValueError:
        return ("", "", "Invalid: could not parse the numbers (use e.g. 3, 7, 9, 50).")

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
    generated = output_ids[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    reasoning_text = split_reasoning(text)
    answer_text = task.extract_answer(text) or ""
    verdict_text = format_verdict(answer_text, nums, target)

    return reasoning_text, answer_text, verdict_text


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
                placeholder="e.g. 3, 7, 9, 50",
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
