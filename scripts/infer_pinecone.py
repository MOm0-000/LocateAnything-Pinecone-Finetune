#!/usr/bin/env python3
"""Run a reproducible LocateAnything pinecone inference probe."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
PROMPT = "Locate all the instances that matches the following description: pinecone."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid")
    parser.add_argument("--attention", choices=("sdpa", "magi"), default="sdpa")
    return parser.parse_args()


def configure_attention(config, implementation: str) -> None:
    config._attn_implementation = implementation
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = implementation
    config.text_config._attn_implementation_autoset = False
    # MoonViT also supports SDPA and does not require flash-attn for this probe.
    config.vision_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation_autoset = False


def main() -> int:
    args = parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    configure_attention(config, args.attention)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        config=config,
    ).to("cuda").eval()
    load_seconds = time.perf_counter() - started

    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    ).to("cuda")

    started = time.perf_counter()
    with torch.no_grad():
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            generation_mode=args.generation_mode,
            temperature=0.01,
            do_sample=True,
            top_p=1.0,
            repetition_penalty=1.0,
            verbose=False,
        )
    infer_seconds = time.perf_counter() - started
    answer = response[0] if isinstance(response, tuple) else response
    boxes = [[int(value) for value in match] for match in BOX_RE.findall(answer)]
    invalid_boxes = [box for box in boxes if box[0] >= box[2] or box[1] >= box[3]]

    result = {
        "model": args.model,
        "image": str(args.image),
        "prompt": PROMPT,
        "attention": args.attention,
        "generation_mode": args.generation_mode,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "answer": answer,
        "box_count": len(boxes),
        "invalid_box_count": len(invalid_boxes),
        "load_seconds": round(load_seconds, 3),
        "infer_seconds": round(infer_seconds, 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
