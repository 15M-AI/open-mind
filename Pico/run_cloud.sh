#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/licensed-corpus.txt [additional Pico train.py options]" >&2
  exit 2
fi

dataset="$1"
shift

if [[ ! -f "${dataset}" || ! -r "${dataset}" ]]; then
  echo "Dataset file is missing or unreadable: ${dataset}" >&2
  exit 2
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use a GPU runner or invoke Pico/train.py with --device cpu.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

python "${script_dir}/prepare_data.py" "${dataset}"
python -m pip install 'numpy>=1.24'
python -m pip install --no-deps -e "${repo_root}/open-mind-train"
python "${script_dir}/train.py" --device cuda --dataset "${dataset}" "$@"
