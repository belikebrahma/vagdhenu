#!/usr/bin/env bash
set -e
# Python 3.10 + a CUDA 12.1 GPU required for inference.
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# NVIDIA BigVGAN is a repo, not a pip package (no setup.py). Clone it and add to PYTHONPATH so
# `import bigvgan` resolves (we use the torch path / use_cuda_kernel=False, so nothing is compiled).
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"   # add this to your shell rc for persistent use
python scripts/download_weights.py                        # our weights -> models/ + IndicF5 base (vocab)
echo "✓ setup complete — see Quickstart in README.md"
