# LocateAnything Pinecone Fine-tuning

本项目用于将 [NVlabs/Eagle](https://github.com/NVlabs/Eagle) 中的 LocateAnything-3B 微调为单类别松果定位模型，并将合并后的模型转换为 GGUF，部署到 Jetson Orin Nano 8GB。

> 这是社区复现实验，不是 NVIDIA 官方仓库。数据集和模型权重不包含在本仓库中。

## 结果

完整留出验证集包含 201 张图片、311 个松果框。所有对比均采用相同样本和 `slow` 解码。

| 运行格式 | 模型 | IoU 阈值 | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Transformers BF16 | 原始 | 0.25 | 0.3735 | 0.2990 | 0.3321 |
| Transformers BF16 | 微调 | 0.25 | 0.7460 | 0.4534 | 0.5640 |
| Transformers BF16 | 原始 | 0.50 | 0.2329 | 0.1865 | 0.2071 |
| Transformers BF16 | 微调 | 0.50 | 0.5397 | 0.3280 | 0.4080 |
| llama.cpp Q4_K_M | 原始 | 0.25 | 0.3120 | 0.2669 | 0.2877 |
| llama.cpp Q4_K_M | 微调 | 0.25 | 0.6535 | 0.4791 | 0.5529 |
| llama.cpp Q4_K_M | 原始 | 0.50 | 0.1992 | 0.1704 | 0.1837 |
| llama.cpp Q4_K_M | 微调 | 0.50 | 0.4825 | 0.3537 | 0.4082 |

这里的 IoU 0.25/0.50 是判定 TP 的固定阈值，不是模型的平均 IoU。完整指标见 [results](results/README.md)。

## 微调方式

- 基座：`NVlabs/LocateAnything-3B`
- 训练数据：558 张，934 个框，包含 247 张负样本
- 方法：仅语言模型 LoRA，rank 32
- 覆盖语言模型 36 层中的 `q/k/v/o_proj` 和 `gate/up/down_proj`
- 视觉骨干和多模态投影层保持冻结
- 训练：558 步，BF16/TF32，SDPA，学习率 `2e-5`
- 输入图像：1600×900；最大序列长度 3072

LoRA 会在导出前合并到语言模型，因此 Jetson 只需文本 GGUF 和 mmproj，不需要单独加载 LoRA adapter。

## 仓库结构

```text
configs/   LocateAnything recipe 示例
deploy/    Jetson runtime 构建和单图推理脚本
docs/      数据、训练、GGUF 和部署说明
patches/   Eagle 在本环境使用 SDPA 的小型补丁
results/   201 张完整评估的汇总与去本机路径原始 JSON
scripts/   转换、训练、评估、合并与可视化脚本
```

## 快速开始

1. 按 [Eagle 官方说明](https://github.com/NVlabs/Eagle/tree/main/Embodied)安装环境并准备完整 LocateAnything-3B 权重。
2. 将 YOLO 数据转换成 LocateAnything ShareGPT JSONL：

```bash
python scripts/convert_yolo_to_locateanything.py --help
```

3. 复制并修改训练 recipe：

```bash
cp configs/recipe.example.json configs/recipe.local.json
```

4. 启动 LoRA 微调：

```bash
META_PATH=/absolute/path/to/recipe.local.json \
REPO_DIR=$HOME/projects/Eagle/Embodied \
MODEL_PATH=$HOME/model-cache/LocateAnything-3B \
OUTPUT_DIR=$HOME/work_dirs/locateanything-pinecone/lora-r32 \
bash scripts/train_pinecone_lora.sh
```

5. 评估、合并和导出步骤见 [训练与导出说明](docs/TRAINING_AND_EXPORT.md)。Jetson 部署见 [Jetson 说明](docs/JETSON.md)。

## 不包含的内容

- 原始或转换后的数据集
- BF16、Q4_K_M GGUF 和 Hugging Face 权重
- 训练 checkpoint、缓存和日志
- 包含数据集图片的错误可视化文件

GitHub 单文件限制也不适合直接托管这些模型文件。`results/summary.json` 中保留了最终产物文件名、大小和 SHA-256，便于核验离线文件。

## 局限

- 现有数据质量有限，原 `test` 被用作留出验证集，不是新的最终盲测集。
- 目前只微调语言模型；小目标、多目标和复杂背景仍有明显漏检。
- ARM64 与 x86-64 可以使用相同 GGUF，但 Jetson 必须使用 ARM64/CUDA 原生编译的 runtime。
- 进入生产或机载集成前，应使用更高质量数据重新训练，并保留从未参与调参的独立测试集。

## 上游与授权

本仓库依赖 Eagle/LocateAnything 和 LocateAnything llama.cpp 分支；它们分别适用各自的上游许可证。详情见 [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md)。本仓库暂未附加独立许可证。
