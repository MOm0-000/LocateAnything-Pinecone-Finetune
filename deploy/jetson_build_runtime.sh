#!/usr/bin/env bash
set -euo pipefail

# Build the LocateAnything-enabled llama.cpp fork natively on a Jetson Orin Nano.
# This script intentionally does not install or replace JetPack/CUDA.

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${PACKAGE_DIR}/runtime/llama.cpp-locateanything"
PINNED_COMMIT="8c1921616abbbbac17493c0d60ae2a73edf7f761"
BUILD_JOBS="${BUILD_JOBS:-2}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "error: this script is intended for the Jetson ARM64 host (aarch64)." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc was not found. Install/repair the JetPack CUDA development components first." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y git cmake build-essential

mkdir -p "$(dirname "${RUNTIME_DIR}")"
if [[ ! -d "${RUNTIME_DIR}/.git" ]]; then
  git clone --branch mtmd-grounders --single-branch \
    https://github.com/yuuko-eth/llama.cpp.git "${RUNTIME_DIR}"
  git -C "${RUNTIME_DIR}" checkout --detach "${PINNED_COMMIT}"
fi

CURRENT_COMMIT="$(git -C "${RUNTIME_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "error: runtime checkout is ${CURRENT_COMMIT}, expected ${PINNED_COMMIT}." >&2
  echo "Remove or move ${RUNTIME_DIR}, then run this script again for a clean pinned clone." >&2
  exit 1
fi

cmake -S "${RUNTIME_DIR}" -B "${RUNTIME_DIR}/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF

cmake --build "${RUNTIME_DIR}/build" \
  --config Release \
  --target llama-mtmd-cli \
  -j "${BUILD_JOBS}"

"${RUNTIME_DIR}/build/bin/llama-mtmd-cli" --version
echo "Runtime build completed."
