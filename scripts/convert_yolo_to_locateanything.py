#!/usr/bin/env python3
"""Convert a YOLO detection dataset to LocateAnything ShareGPT JSONL.

The converter never modifies the source dataset.  It creates a new, self-contained
output directory with copied (or hard-linked) images, one JSONL file per split, a
training recipe, and a machine-readable conversion report.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROMPT_TEMPLATE = (
    "Locate all the instances that matches the following description: {classes}."
)
COORD_TOKEN_RE = re.compile(r"<(\d+)>")


class ConversionError(RuntimeError):
    """Raised when conversion cannot safely continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO bounding boxes to LocateAnything JSONL + recipe JSON."
    )
    parser.add_argument("--source", type=Path, required=True, help="YOLO dataset root")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--config",
        type=Path,
        help="YOLO data YAML/TXT (auto-detected when omitted)",
    )
    parser.add_argument(
        "--dataset-name",
        help="Recipe/annotation name (default: class name for one-class datasets)",
    )
    parser.add_argument(
        "--wsl-output-root",
        help="Absolute WSL path of --output, used in recipe JSON",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val"),
        help="Dataset splits to convert (default: train val)",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How images are placed in output (default: copy)",
    )
    parser.add_argument(
        "--empty-policy",
        choices=("skip", "negative", "error"),
        default="skip",
        help=(
            "How empty label files are handled. 'skip' avoids treating possible "
            "missing annotations as negative samples (default: skip)."
        ),
    )
    parser.add_argument(
        "--no-data-augment",
        action="store_true",
        help="Set data_augment=false in the training recipe",
    )
    return parser.parse_args()


def find_config(source: Path, explicit: Path | None) -> Path:
    if explicit:
        path = explicit.resolve()
        if not path.is_file():
            raise ConversionError(f"Config file does not exist: {path}")
        return path
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "data.txt"):
        candidate = source / name
        if candidate.is_file():
            return candidate
    raise ConversionError(
        "Could not find data.yaml, data.yml, dataset.yaml, dataset.yml, or data.txt"
    )


def parse_inline_value(raw: str) -> Any:
    value = raw.strip()
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def load_simple_yolo_config(path: Path) -> dict[str, Any]:
    """Read the flat YAML subset used by this dataset without a PyYAML dependency."""
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line or line.startswith("-"):
            continue
        key, raw_value = line.split(":", 1)
        if raw_value.strip():
            result[key.strip()] = parse_inline_value(raw_value)

    names = result.get("names")
    if isinstance(names, dict):
        try:
            names = [names[key] for key in sorted(names, key=lambda item: int(item))]
        except (TypeError, ValueError):
            raise ConversionError("Class-name dictionary keys must be integer class IDs")
    if not isinstance(names, (list, tuple)) or not names:
        raise ConversionError(
            f"{path} must contain a one-line class list, e.g. names: ['pinecone']"
        )
    class_names = [str(name).strip() for name in names]
    if any(not name for name in class_names):
        raise ConversionError("Class names must not be empty")

    nc = result.get("nc", len(class_names))
    try:
        nc = int(nc)
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"Invalid nc value in {path}: {nc!r}") from exc
    if nc != len(class_names):
        raise ConversionError(f"nc={nc}, but names contains {len(class_names)} classes")

    result["names"] = class_names
    result["nc"] = nc
    return result


def sanitize_name(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip()).strip("_.-")
    return result or "detection"


def windows_path_to_wsl(path: Path) -> str:
    resolved = str(path.resolve())
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", resolved)
    if match:
        drive, rest = match.groups()
        return f"/mnt/{drive.lower()}/{rest.replace(os.sep, '/')}"
    return resolved.replace(os.sep, "/")


def split_image_dir(source: Path, config: dict[str, Any], split: str) -> Path:
    configured = config.get(split)
    if configured:
        configured_path = Path(str(configured))
        if configured_path.is_absolute():
            return configured_path.resolve()
        return (source / configured_path).resolve()
    return (source / "images" / split).resolve()


def split_label_dir(source: Path, image_dir: Path, split: str) -> Path:
    try:
        relative = image_dir.relative_to(source)
    except ValueError:
        return (source / "labels" / split).resolve()
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return (source.joinpath(*parts)).resolve()
    return (source / "labels" / split).resolve()


def quantize_coordinate(value: float) -> int:
    # Explicit half-up rounding avoids Python's bankers-rounding behavior.
    return min(1000, max(0, int(math.floor(value * 1000.0 + 0.5))))


def read_yolo_label(
    label_path: Path, class_names: list[str]
) -> tuple[list[tuple[int, int, int, int, int]], list[str]]:
    boxes: list[tuple[int, int, int, int, int]] = []
    errors: list[str] = []
    raw_lines = label_path.read_text(encoding="utf-8-sig").splitlines()

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(
                f"line {line_number}: expected 5 fields, found {len(fields)}"
            )
            continue
        try:
            class_value, cx, cy, width, height = map(float, fields)
        except ValueError:
            errors.append(f"line {line_number}: contains a non-numeric value")
            continue
        if not all(math.isfinite(value) for value in (class_value, cx, cy, width, height)):
            errors.append(f"line {line_number}: contains NaN or infinity")
            continue
        if not class_value.is_integer():
            errors.append(f"line {line_number}: class ID must be an integer")
            continue
        class_id = int(class_value)
        if class_id < 0 or class_id >= len(class_names):
            errors.append(f"line {line_number}: class ID {class_id} is out of range")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: width and height must be positive")
            continue

        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0
        tolerance = 1e-6
        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > 1.0 + tolerance
            or y2 > 1.0 + tolerance
        ):
            errors.append(f"line {line_number}: bounding box extends outside [0, 1]")
            continue
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        qx1, qy1, qx2, qy2 = map(
            quantize_coordinate, (x1, y1, x2, y2)
        )
        if qx1 >= qx2 or qy1 >= qy2:
            errors.append(
                f"line {line_number}: box collapses after normalization to 0..1000"
            )
            continue
        boxes.append((class_id, qx1, qy1, qx2, qy2))

    return boxes, errors


def build_response(
    boxes: list[tuple[int, int, int, int, int]], class_names: list[str]
) -> str:
    if not boxes:
        return "<box>none</box>"
    # Deterministic class-first, top-to-bottom, left-to-right ordering.
    ordered = sorted(boxes, key=lambda item: (item[0], item[2], item[1], item[4], item[3]))
    return "".join(
        f"<ref>{class_names[class_id]}</ref>"
        f"<box><{x1}><{y1}><{x2}><{y2}></box>"
        for class_id, x1, y1, x2, y2 in ordered
    )


def place_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def validate_jsonl(path: Path, image_root: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if set(item) != {"conversations", "image"}:
                raise ConversionError(f"Unexpected fields at {path}:{line_number}")
            turns = item["conversations"]
            if (
                not isinstance(turns, list)
                or len(turns) != 2
                or turns[0].get("from") != "human"
                or turns[1].get("from") != "gpt"
            ):
                raise ConversionError(f"Invalid ShareGPT conversation at {path}:{line_number}")
            image_path = image_root.joinpath(*PurePosixPath(item["image"]).parts)
            if not image_path.is_file():
                raise ConversionError(
                    f"Missing referenced image at {path}:{line_number}: {item['image']}"
                )
            answer = turns[1].get("value", "")
            coordinates = [int(value) for value in COORD_TOKEN_RE.findall(answer)]
            if any(value < 0 or value > 1000 for value in coordinates):
                raise ConversionError(f"Coordinate outside 0..1000 at {path}:{line_number}")
            if answer != "<box>none</box>" and not re.fullmatch(
                r"(?:<ref>[^<>]+</ref><box><\d+><\d+><\d+><\d+></box>)+",
                answer,
            ):
                raise ConversionError(f"Invalid LocateAnything answer at {path}:{line_number}")
            count += 1
    return count


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise ConversionError(f"Source directory does not exist: {source}")
    if output.exists():
        raise ConversionError(
            f"Output already exists: {output}. Refusing to overwrite it."
        )
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ConversionError("Output must not be inside the source dataset")

    config_path = find_config(source, args.config)
    config = load_simple_yolo_config(config_path)
    class_names: list[str] = config["names"]
    default_name = class_names[0] if len(class_names) == 1 else "yolo_detection"
    dataset_name = sanitize_name(args.dataset_name or default_name)
    wsl_output_root = (args.wsl_output_root or windows_path_to_wsl(output)).rstrip("/")

    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    if temporary.exists():
        raise ConversionError(f"Temporary output unexpectedly exists: {temporary}")
    temporary.mkdir(parents=True)

    report: dict[str, Any] = {
        "format": "LocateAnything ShareGPT JSONL",
        "coordinate_range": [0, 1000],
        "source": str(source),
        "source_config": str(config_path),
        "output": str(output),
        "wsl_output_root": wsl_output_root,
        "classes": class_names,
        "empty_policy": args.empty_policy,
        "requested_image_mode": args.image_mode,
        "splits": {},
        "skipped_samples": [],
        "unused_labels": [],
    }

    try:
        annotations_dir = temporary / "annotations"
        images_output_root = temporary / "images"
        annotations_dir.mkdir()
        images_output_root.mkdir()
        converted_splits: list[str] = []

        for split in args.splits:
            image_dir = split_image_dir(source, config, split)
            label_dir = split_label_dir(source, image_dir, split)
            if not image_dir.is_dir():
                report["splits"][split] = {
                    "status": "not_found",
                    "image_dir": str(image_dir),
                }
                continue
            if not label_dir.is_dir():
                raise ConversionError(f"Label directory does not exist: {label_dir}")

            image_paths = sorted(
                (
                    path
                    for path in image_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: path.relative_to(image_dir).as_posix(),
            )
            label_paths = {
                path.relative_to(label_dir).with_suffix("").as_posix(): path
                for path in label_dir.rglob("*.txt")
                if path.is_file()
            }
            used_label_keys: set[str] = set()
            class_counts: Counter[str] = Counter()
            output_modes: Counter[str] = Counter()
            annotation_path = annotations_dir / f"{dataset_name}_{split}.jsonl"
            samples_written = 0
            boxes_written = 0
            empty_labels = 0
            missing_labels = 0
            invalid_samples = 0

            prompt_classes = "</c>".join(class_names)
            prompt = PROMPT_TEMPLATE.format(classes=prompt_classes)
            with annotation_path.open("w", encoding="utf-8", newline="\n") as writer:
                for image_path in image_paths:
                    relative_image = image_path.relative_to(image_dir)
                    label_key = relative_image.with_suffix("").as_posix()
                    label_path = label_paths.get(label_key)
                    if label_path is None:
                        missing_labels += 1
                        report["skipped_samples"].append(
                            {
                                "split": split,
                                "image": str(image_path),
                                "reason": "missing_label_file",
                            }
                        )
                        continue
                    used_label_keys.add(label_key)
                    boxes, errors = read_yolo_label(label_path, class_names)
                    if errors:
                        invalid_samples += 1
                        report["skipped_samples"].append(
                            {
                                "split": split,
                                "image": str(image_path),
                                "label": str(label_path),
                                "reason": "invalid_label",
                                "details": errors,
                            }
                        )
                        continue
                    if not boxes:
                        empty_labels += 1
                        if args.empty_policy == "error":
                            raise ConversionError(f"Empty label file: {label_path}")
                        if args.empty_policy == "skip":
                            report["skipped_samples"].append(
                                {
                                    "split": split,
                                    "image": str(image_path),
                                    "label": str(label_path),
                                    "reason": "empty_label_skipped",
                                }
                            )
                            continue

                    destination = images_output_root / split / relative_image
                    output_modes[place_image(image_path, destination, args.image_mode)] += 1
                    image_field = (PurePosixPath(split) / PurePosixPath(relative_image.as_posix())).as_posix()
                    item = {
                        "conversations": [
                            {"from": "human", "value": prompt},
                            {"from": "gpt", "value": build_response(boxes, class_names)},
                        ],
                        "image": image_field,
                    }
                    writer.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                    writer.write("\n")
                    samples_written += 1
                    boxes_written += len(boxes)
                    for class_id, *_ in boxes:
                        class_counts[class_names[class_id]] += 1

            for label_key, label_path in sorted(label_paths.items()):
                if label_key not in used_label_keys:
                    report["unused_labels"].append(
                        {"split": split, "label": str(label_path), "reason": "no_matching_image"}
                    )

            validated_samples = validate_jsonl(annotation_path, images_output_root)
            if validated_samples != samples_written:
                raise ConversionError(
                    f"Validation count mismatch for {split}: "
                    f"wrote {samples_written}, read {validated_samples}"
                )
            report["splits"][split] = {
                "status": "converted",
                "source_images": len(image_paths),
                "samples_written": samples_written,
                "boxes_written": boxes_written,
                "class_box_counts": dict(sorted(class_counts.items())),
                "empty_labels": empty_labels,
                "missing_labels": missing_labels,
                "invalid_samples": invalid_samples,
                "image_placement": dict(sorted(output_modes.items())),
                "annotation": str(output / "annotations" / annotation_path.name),
            }
            converted_splits.append(split)

        if "train" not in converted_splits:
            raise ConversionError("A converted train split is required to create the recipe")
        if report["splits"]["train"]["samples_written"] == 0:
            raise ConversionError("The train split contains no usable samples")

        recipe = {
            f"{dataset_name}_train": {
                "annotation": f"{wsl_output_root}/annotations/{dataset_name}_train.jsonl",
                "root": f"{wsl_output_root}/images/",
                "repeat_time": 1.0,
                "data_augment": not args.no_data_augment,
            }
        }
        (temporary / "recipe_train.json").write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["recipe"] = str(output / "recipe_train.json")
        report["validation"] = "passed"
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        readme = f"""LocateAnything dataset converted from YOLO

Source (not modified): {source}
Classes: {', '.join(class_names)}
Coordinates: normalized integer tokens in [0, 1000]
Empty-label policy: {args.empty_policy}

Files:
  annotations/{dataset_name}_train.jsonl  training annotations
  annotations/{dataset_name}_val.jsonl    validation annotations (when present)
  images/train and images/val             self-contained image copies/links
  recipe_train.json                       pass to training with --meta_path
  conversion_report.json                  counts, skipped samples, and validation

WSL recipe path:
  {wsl_output_root}/recipe_train.json
"""
        (temporary / "README.txt").write_text(readme, encoding="utf-8")
        temporary.replace(output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        report = convert(args)
    except (ConversionError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        split: {
            "samples": values.get("samples_written", 0),
            "boxes": values.get("boxes_written", 0),
            "empty_labels": values.get("empty_labels", 0),
            "invalid_samples": values.get("invalid_samples", 0),
        }
        for split, values in report["splits"].items()
    }
    print(json.dumps({"status": "ok", "splits": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
