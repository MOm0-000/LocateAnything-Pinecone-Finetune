#!/usr/bin/env python3
"""Render side-by-side ground-truth and prediction views for failed images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
HEADER_HEIGHT = 64
DIVIDER_WIDTH = 6
GT_COLOR = (0, 255, 0)
PRED_COLOR = (255, 55, 55)


def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    return [tuple(map(int, match)) for match in BOX_RE.findall(text)]


def to_pixels(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1 * width / 1000))),
        max(0, min(height - 1, round(y1 * height / 1000))),
        max(0, min(width - 1, round(x2 * width / 1000))),
        max(0, min(height - 1, round(y2 * height / 1000))),
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if path.exists():
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def render_panel(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int],
    title: str,
    prefix: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + HEADER_HEIGHT), "black")
    panel.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text((18, 15), title, fill="white", font=font)
    line_width = max(4, round(min(image.size) / 220))

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = to_pixels(box, image.width, image.height)
        shifted = (x1, y1 + HEADER_HEIGHT, x2, y2 + HEADER_HEIGHT)
        draw.rectangle(shifted, outline=color, width=line_width)
        label_y = max(HEADER_HEIGHT, shifted[1] - 30)
        draw.text(
            (shifted[0] + 3, label_y),
            f"{prefix}{index}",
            fill=color,
            font=font,
            stroke_width=3,
            stroke_fill="black",
        )
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    annotations = {}
    for line in args.annotation.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            annotations[row["image"]] = parse_boxes(row["conversations"][1]["value"])

    failed = [
        record
        for record in result["records"]
        if record["matches"]["0.5"]["fp"] or record["matches"]["0.5"]["fn"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    font = load_font(27)

    for position, record in enumerate(failed, start=1):
        relative_image = record["image"]
        image_path = args.image_root / relative_image
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        gt_boxes = annotations[relative_image]
        pred_boxes = parse_boxes(record["answer"])
        metrics = record["matches"]["0.5"]
        left = render_panel(
            image,
            gt_boxes,
            GT_COLOR,
            f"{image_path.name} | Ground truth: {len(gt_boxes)} (green)",
            "GT",
            font,
        )
        right = render_panel(
            image,
            pred_boxes,
            PRED_COLOR,
            (
                f"Prediction: {len(pred_boxes)} (red) | IoU 0.50 "
                f"TP/FP/FN: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
            ),
            "P",
            font,
        )

        comparison = Image.new(
            "RGB",
            (left.width + DIVIDER_WIDTH + right.width, left.height),
            "white",
        )
        comparison.paste(left, (0, 0))
        comparison.paste(right, (left.width + DIVIDER_WIDTH, 0))
        output_path = args.output / f"{image_path.stem}_comparison.jpg"
        comparison.save(output_path, quality=92, subsampling=0, optimize=True)
        print(f"[{position}/{len(failed)}] {output_path}")

    print(f"Rendered {len(failed)} comparison images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
