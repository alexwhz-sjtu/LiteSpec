# Source this file from Bash to activate LiteSpec with its CUDA 12.8 compiler.
_litespec_environment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_litespec_environment_root}/.venv/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/root/.local/bin:${CUDA_HOME}/bin:${PATH}"
unset _litespec_environment_root
