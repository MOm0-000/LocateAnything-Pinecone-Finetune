#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/image.jpg" >&2
  exit 2
fi

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_BIN="${RUNTIME_BIN:-${PACKAGE_DIR}/runtime/llama.cpp-locateanything/build/bin/llama-mtmd-cli}"
MODEL_FILE="${MODEL_FILE:-${PACKAGE_DIR}/LocateAnything-Pinecone-Second-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-${PACKAGE_DIR}/mmproj-LocateAnything-Pinecone-Second-BF16.gguf}"
IMAGE_FILE="$1"

GROUNDING_MODE="${MTMD_GROUNDING_MODE:-slow}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_PREDICT="${N_PREDICT:-512}"
THREADS="${THREADS:-6}"

case "${GROUNDING_MODE}" in
  slow|hybrid|fast) ;;
  *) echo "error: MTMD_GROUNDING_MODE must be slow, hybrid, or fast." >&2; exit 2 ;;
esac

for REQUIRED_FILE in "${RUNTIME_BIN}" "${MODEL_FILE}" "${MMPROJ_FILE}" "${IMAGE_FILE}"; do
  if [[ ! -f "${REQUIRED_FILE}" ]]; then
    echo "error: missing file: ${REQUIRED_FILE}" >&2
    exit 1
  fi
done

export MTMD_GROUNDING_MODE="${GROUNDING_MODE}"
exec "${RUNTIME_BIN}" \
  -m "${MODEL_FILE}" \
  --mmproj "${MMPROJ_FILE}" \
  --image "${IMAGE_FILE}" \
  -p "Locate all the instances that matches the following description: pinecone." \
  -c "${CTX_SIZE}" \
  -n "${N_PREDICT}" \
  -t "${THREADS}" \
  --temp 0 \
  --flash-attn auto \
  --fit on \
  --fit-target 768 \
  -ngl all \
  --mmproj-offload \
  --no-warmup
