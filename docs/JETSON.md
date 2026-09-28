# Jetson Orin Nano 8GB 部署

GGUF 权重不绑定 x86-64 或 ARM64。需要按 Jetson ARM64/CUDA 环境重新编译的是推理程序，不是模型。

## 文件

将以下文件自行复制到 Jetson；本仓库不包含模型权重：

- `LocateAnything-Pinecone-Second-Q4_K_M.gguf`
- `mmproj-LocateAnything-Pinecone-Second-BF16.gguf`
- `deploy/jetson_build_runtime.sh`
- `deploy/jetson_run_pinecone.sh`

模型 SHA-256 位于 `results/summary.json`。

## 构建 runtime

```bash
chmod +x jetson_build_runtime.sh jetson_run_pinecone.sh
BUILD_JOBS=2 ./jetson_build_runtime.sh
```

脚本固定使用 LocateAnything `llama.cpp` 分支与提交，并为 Orin 的 CUDA SM87 构建 `llama-mtmd-cli`。要求 JetPack/CUDA 开发组件完整且 `nvcc` 可用。

## 单图推理

```bash
./jetson_run_pinecone.sh /absolute/path/to/image.jpg
```

默认参数为 `slow` 解码、上下文 4096、最大输出 512、温度 0。可通过环境变量覆盖：

```bash
CTX_SIZE=3072 N_PREDICT=256 THREADS=6 \
./jetson_run_pinecone.sh /absolute/path/to/image.jpg
```

如果已有编译好的 LocateAnything runtime：

```bash
RUNTIME_BIN=/path/to/llama-mtmd-cli \
MODEL_FILE=/path/to/LocateAnything-Pinecone-Second-Q4_K_M.gguf \
MMPROJ_FILE=/path/to/mmproj-LocateAnything-Pinecone-Second-BF16.gguf \
./jetson_run_pinecone.sh /absolute/path/to/image.jpg
```

8GB 为统一内存。出现内存不足时，先降低上下文和最大输出；必要时减少 GPU offload。实机验收应同时记录 `tegrastats`、单帧耗时、温度和持续运行稳定性。
