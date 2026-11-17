#!/usr/bin/env bash
# Build a self-contained Python 3.10 + torch 2.4.1/cu124 environment with micromamba.
# Works on Colab (whatever system Python/torch it ships), RunPod and any Linux box with an
# NVIDIA driver >= 12.4. All heavy deps come as prebuilt wheels (no compilation).
#
#   bash colab_env.sh /content/env          # ~8-12 min on Colab
#   /content/env/bin/python -c "import spconv.pytorch, flash_attn, torch_scatter; print('ok')"
set -euo pipefail
ENV=${1:-/content/env}
TORCH=2.4.1
TV=0.19.1
CU=cu124

if [ ! -x "$ENV/bin/python" ]; then
  if ! command -v micromamba >/dev/null 2>&1; then
    mkdir -p /opt/mm && cd /opt/mm
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
    export PATH=/opt/mm/bin:$PATH
  fi
  micromamba create -y -q -p "$ENV" -c conda-forge python=3.10 pip git ninja
fi
PIP="$ENV/bin/python -m pip"
$PIP install -q --upgrade pip wheel setuptools packaging psutil
$PIP install -q torch==$TORCH torchvision==$TV --index-url https://download.pytorch.org/whl/$CU
$PIP install -q torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+$CU.html
$PIP install -q spconv-$CU
# flash-attn: setup.py downloads the matching prebuilt wheel (torch 2.4 / cu12 / cp310 exists)
MAX_JOBS=4 $PIP install -q "flash-attn==2.7.4.post1" --no-build-isolation
$PIP install -q "numpy<2.3" scipy addict timm einops plyfile termcolor colorhash tensorboard
"$ENV/bin/python" - << 'EOF'
import torch, spconv.pytorch, torch_scatter, flash_attn
print("env ok | torch", torch.__version__, "| cuda", torch.version.cuda, "| spconv", spconv.__version__,
      "| flash_attn", flash_attn.__version__, "| gpu", torch.cuda.is_available())
EOF
