from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import detect_default_device, parse_safety_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prompt/response moderation inference with an exported HF model."
    )
    parser.add_argument("--model-path", required=True, help="Path to exported HF model")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--response", default=None, help="Optional assistant response text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default=detect_default_device())
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    if args.response is not None:
        messages.append({"role": "assistant", "content": args.response})

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=not args.disable_thinking,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    generated_ids = generated[:, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    predicted_label = parse_safety_label(output_text)

    print(f"Prediction:\n{output_text}")
    print(f"Parsed safety label: {predicted_label}")


if __name__ == "__main__":
    main()
