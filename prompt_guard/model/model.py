from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ModelConfig:
    model_name_or_path: str
    bf16: bool = True
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None


def _probe_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, bool]:
    """Returns (model, tokenizer, use_fa2)."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    dtype = torch.bfloat16 if config.bf16 else torch.float32

    use_fa2 = _probe_flash_attention()

    if use_fa2:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            print("Attention implementation: flash_attention_2")
        except (ImportError, ValueError) as e:
            print(f"Flash Attention 2 unavailable ({e}), falling back to eager")
            use_fa2 = False

    if not use_fa2:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        print("Attention implementation: eager (block-diagonal mask will be used for packing)")

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = config.lora_target_modules or ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer, use_fa2
