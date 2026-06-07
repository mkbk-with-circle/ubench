#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <benchmark-name> [benchmark args...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$1"
shift

EXE="${ROOT_DIR}/build/bin/${BENCH}"
OUT="${ROOT_DIR}/results/msprof_${BENCH}"

if [[ ! -x "${EXE}" ]]; then
  echo "benchmark executable not found: ${EXE}" >&2
  exit 1
fi

mkdir -p "${OUT}"

msprof op --output="${OUT}" "${EXE}" "$@"

echo "msprof output: ${OUT}"
