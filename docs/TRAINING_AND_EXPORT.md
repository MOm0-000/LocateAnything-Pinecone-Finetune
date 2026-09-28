# 训练、评估与 GGUF 导出

## 1. 准备上游代码

```bash
git clone https://github.com/NVlabs/Eagle.git
cd Eagle
git checkout 783f656d127ee498137b5ff52603ce36c292d317
git apply /path/to/LocateAnything-Pinecone-Finetune/patches/eagle_wsl_sdpa.patch
```

Python/CUDA 依赖应优先遵循 Eagle 官方 Embodied 文档。补丁用于本实验环境的 SDPA 与分布式后端兼容，不保证适合所有 CUDA/PyTorch 组合。

## 2. LoRA 微调

```bash
META_PATH=/absolute/path/to/recipe.json \
REPO_DIR=$HOME/projects/Eagle/Embodied \
VENV_DIR=$HOME/venvs/locateanything \
MODEL_PATH=$HOME/model-cache/LocateAnything-3B \
OUTPUT_DIR=$HOME/work_dirs/locateanything-pinecone/lora-r32 \
MAX_STEPS=558 GRADIENT_ACC=1 MAX_SEQ_LENGTH=3072 MAX_NUM_TOKENS=3072 \
PACKING_BUFFER_SIZE=2 bash scripts/train_pinecone_lora.sh
```

该启动器冻结视觉骨干和多模态投影层，只在语言模型全部 36 层的注意力与 MLP 线性层上添加 rank-32 LoRA。

## 3. Transformers 评估

```bash
python scripts/evaluate_pinecone.py \
  --model $HOME/work_dirs/locateanything-pinecone/lora-r32 \
  --annotation /path/to/pinecone_val.jsonl \
  --image-root /path/to/dataset_true/images \
  --output results/eval_finetuned_transformers.json \
  --limit 0 --seed 42 --max-new-tokens 512 \
  --generation-mode slow --attention sdpa
```

## 4. 合并 LoRA

```bash
python scripts/merge_locateanything_lora.py \
  --input $HOME/work_dirs/locateanything-pinecone/lora-r32 \
  --output $HOME/work_dirs/locateanything-pinecone/lora-r32-merged
```

合并只作用于嵌套的 `language_model`；视觉编码器和投影器保持原权重。

## 5. 转换和量化

使用 LocateAnything llama.cpp 分支的转换器：

```bash
python /path/to/llama.cpp-locateanything/convert_hf_to_gguf.py \
  $HOME/work_dirs/locateanything-pinecone/lora-r32-merged \
  --outfile LocateAnything-Pinecone-BF16.gguf --outtype bf16

python /path/to/llama.cpp-locateanything/convert_hf_to_gguf.py \
  $HOME/work_dirs/locateanything-pinecone/lora-r32-merged \
  --outfile mmproj-LocateAnything-Pinecone-BF16.gguf \
  --outtype bf16 --mmproj

/path/to/llama.cpp-locateanything/build/bin/llama-quantize \
  LocateAnything-Pinecone-BF16.gguf \
  LocateAnything-Pinecone-Q4_K_M.gguf Q4_K_M
```

为文本 GGUF 和 mmproj 指定不同输出路径，避免互相覆盖。量化后再用 `evaluate_gguf_pinecone.py` 对原始与微调模型做同口径完整评估。
