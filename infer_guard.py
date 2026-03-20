import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
CATEGORIES_PATTERN = re.compile(
    r"Categories:\s*(.+?)(?:\n|$)"
)
REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)")


def parse_output(text: str) -> dict:
    safety_match = SAFETY_PATTERN.search(text)
    categories_match = CATEGORIES_PATTERN.search(text)
    refusal_match = REFUSAL_PATTERN.search(text)

    categories = None
    if categories_match:
        raw = categories_match.group(1).strip()
        categories = [part.strip() for part in raw.split(",")] if raw else []

    return {
        "safety": safety_match.group(1) if safety_match else None,
        "categories": categories,
        "refusal": refusal_match.group(1) if refusal_match else None,
        "raw_text": text.strip(),
    }


def build_messages(prompt: str, response: str | None) -> list[dict]:
    messages = [{"role": "user", "content": prompt}]
    if response:
        messages.append({"role": "assistant", "content": response})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Run inference with exported generative guard model.")
    parser.add_argument("--model-path", required=True, help="Directory with save_pretrained model")
    parser.add_argument("--prompt", required=True, help="User prompt to moderate")
    parser.add_argument("--response", default=None, help="Optional assistant response to moderate")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda, cpu, mps, ...",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    args = parser.parse_args()

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

    messages = build_messages(args.prompt, args.response)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = generated[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    result = parse_output(output_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
