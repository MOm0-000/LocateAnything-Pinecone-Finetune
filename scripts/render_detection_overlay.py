#!/usr/bin/env python3
"""Render LocateAnything predictions and ShareGPT JSONL ground truth on an image."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def boxes(text: str) -> list[tuple[int, int, int, int]]:
    return [tuple(map(int, match)) for match in BOX_RE.findall(text)]


def load_ground_truth(annotation: Path, image_name: str) -> list[tuple[int, int, int, int]]:
    with annotation.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if Path(row["image"]).name == image_name:
                return boxes(row["conversations"][1]["value"])
    raise ValueError(f"No annotation found for {image_name}")


def to_pixels(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    pred_boxes = boxes(prediction["answer"])
    gt_boxes = load_ground_truth(args.annotation, args.image.name)

    for index, box in enumerate(pred_boxes, start=1):
        pixel_box = to_pixels(box, *image.size)
        if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
            color = "orange"
        else:
            color = "red"
        draw.rectangle(pixel_box, outline=color, width=3)
        draw.text((pixel_box[0] + 2, pixel_box[1] + 2), f"P{index}", fill=color, font=font,
                  stroke_width=2, stroke_fill="white")

    for index, box in enumerate(gt_boxes, start=1):
        pixel_box = to_pixels(box, *image.size)
        draw.rectangle(pixel_box, outline="lime", width=5)
        draw.text((pixel_box[0] + 2, max(0, pixel_box[1] - 25)), f"GT{index}", fill="lime", font=font,
                  stroke_width=2, stroke_fill="black")

    legend = f"red: prediction ({len(pred_boxes)})   green: annotation ({len(gt_boxes)})"
    draw.rectangle((0, 0, min(image.width, 760), 38), fill="black")
    draw.text((8, 7), legend, fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
