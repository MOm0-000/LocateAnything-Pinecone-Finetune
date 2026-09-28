#!/usr/bin/env python3
"""Evaluate LocateAnything on a deterministic sample of ShareGPT JSONL data."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(text: str) -> list[list[int]]:
    return [[int(value) for value in match] for match in BOX_RE.findall(text)]


def configure_attention(config, implementation: str) -> None:
    config._attn_implementation = implementation
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = implementation
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation_autoset = False


def box_iou(a: list[int], b: list[int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_boxes(predicted: list[list[int]], target: list[list[int]], threshold: float) -> tuple[int, int, int]:
    candidates = sorted(
        ((box_iou(pred, gt), pi, gi) for pi, pred in enumerate(predicted) for gi, gt in enumerate(target)),
        reverse=True,
    )
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for iou, pi, gi in candidates:
        if iou < threshold:
            break
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
    true_positive = len(matched_pred)
    return true_positive, len(predicted) - true_positive, len(target) - true_positive


def summarize(counts: tuple[int, int, int]) -> dict[str, float | int]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid")
    parser.add_argument("--attention", choices=("sdpa", "magi"), default="sdpa")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.annotation.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit <= 0 or args.limit > len(rows):
        selected = list(enumerate(rows))
    else:
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.limit))
        selected = [(index, rows[index]) for index in indices]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
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
    load_seconds = time.perf_counter() - load_started

    totals = {0.25: [0, 0, 0], 0.5: [0, 0, 0]}
    records = []
    started = time.perf_counter()
    for position, (source_index, row) in enumerate(selected, start=1):
        image_path = args.image_root / row["image"]
        prompt = row["conversations"][0]["value"]
        target_text = row["conversations"][1]["value"]
        target_boxes = parse_boxes(target_text)
        item_started = time.perf_counter()
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
            torch.manual_seed(args.seed + source_index)
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
            answer = response[0] if isinstance(response, tuple) else response
            predicted_boxes = parse_boxes(answer)
            valid_boxes = [box for box in predicted_boxes if box[0] < box[2] and box[1] < box[3]]
            item_matches = {}
            for threshold in totals:
                counts = match_boxes(valid_boxes, target_boxes, threshold)
                totals[threshold] = [left + right for left, right in zip(totals[threshold], counts)]
                item_matches[str(threshold)] = summarize(counts)
            record = {
                "source_index": source_index,
                "image": row["image"],
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "target_box_count": len(target_boxes),
                "predicted_box_count": len(predicted_boxes),
                "invalid_box_count": len(predicted_boxes) - len(valid_boxes),
                "matches": item_matches,
                "answer": answer,
                "seconds": round(time.perf_counter() - item_started, 3),
            }
        except Exception as error:
            record = {
                "source_index": source_index,
                "image": row["image"],
                "target_box_count": len(target_boxes),
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.perf_counter() - item_started, 3),
            }
        records.append(record)
        print(
            f"[{position}/{len(selected)}] {row['image']} gt={len(target_boxes)} "
            f"pred={record.get('predicted_box_count', 'error')} {record['seconds']}s",
            file=sys.stderr,
            flush=True,
        )

    valid_records = [record for record in records if "error" not in record]
    result = {
        "model": args.model,
        "annotation": str(args.annotation),
        "sample_seed": args.seed,
        "requested_samples": len(selected),
        "successful_samples": len(valid_records),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "total_target_boxes": sum(record.get("target_box_count", 0) for record in valid_records),
        "total_predicted_boxes": sum(record.get("predicted_box_count", 0) for record in valid_records),
        "total_invalid_boxes": sum(record.get("invalid_box_count", 0) for record in valid_records),
        "metrics_note": "Reference only: incomplete/noisy annotations can count plausible detections as false positives.",
        "metrics": {str(threshold): summarize(tuple(counts)) for threshold, counts in totals.items()},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if len(valid_records) == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
