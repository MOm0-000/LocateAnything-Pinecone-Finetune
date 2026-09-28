#!/usr/bin/env python3
"""Merge the nested LocateAnything language-model LoRA into BF16 base weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


INFERENCE_FILES = (
    "attn_mask_utils.py",
    "chat_template.json",
    "configuration_locateanything.py",
    "configuration_qwen2.py",
    "generate_utils.py",
    "image_processing_locateanything.py",
    "mask_magi_utils.py",
    "mask_sdpa_utils.py",
    "modeling_locateanything.py",
    "modeling_qwen2.py",
    "modeling_vit.py",
    "processing_locateanything.py",
)


def configure_attention(config) -> None:
    config._attn_implementation = "sdpa"
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation_autoset = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    configure_attention(config)
    if not getattr(config, "use_llm_lora", 0):
        raise ValueError("Input config does not declare use_llm_lora; refusing a no-op merge")

    model = AutoModel.from_pretrained(
        args.input,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        config=config,
        low_cpu_mem_usage=True,
    )
    if not hasattr(model.language_model, "merge_and_unload"):
        raise TypeError(f"Expected a PEFT language model, got {type(model.language_model).__name__}")

    model.language_model = model.language_model.merge_and_unload(progressbar=True)
    model.config.use_llm_lora = 0
    model.config.text_config = model.language_model.config
    configure_attention(model.config)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.input, trust_remote_code=True, use_fast=False)
    tokenizer.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    for name in INFERENCE_FILES:
        source = args.input / name
        if source.is_file():
            shutil.copy2(source, args.output / name)

    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "merged_component": "language_model LoRA",
        "dtype": "bfloat16",
        "use_llm_lora": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
