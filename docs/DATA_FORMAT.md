# 数据格式

转换脚本期望标准单类别 YOLO 结构：

```text
dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

每行 YOLO 标签为：

```text
class_id center_x center_y width height
```

坐标使用 0–1 归一化值。松果类别统一映射为类别 0；无目标图片应保留空标签文件。

运行：

```bash
python scripts/convert_yolo_to_locateanything.py \
  --source /absolute/path/to/yolo_dataset \
  --output /absolute/path/to/dataset_true \
  --dataset-name pinecone \
  --wsl-output-root /absolute/path/to/dataset_true \
  --empty-policy negative \
  --no-data-augment
```

只有确认空标签图片确实为负样本时，才应使用 `--empty-policy negative`；否则保留默认的 `skip`。请先用 `--help` 核对当前脚本参数。输出包括：

- `annotations/*.jsonl`：LocateAnything ShareGPT 风格标注
- `images/`：训练/验证图片
- `recipe_train.json`：Eagle 训练 recipe
- `conversion_report.json`：转换统计与异常记录

正式训练前应检查：图片可读性、无效框、类别映射、空标签是否真为负样本，以及所有样本的视觉 token 数是否低于训练上限。
