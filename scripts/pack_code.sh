#!/usr/bin/env bash
# Pack the code (no data) for Colab: upload data/gridnet_code.zip to Drive GridNet/code/
set -e
cd "$(dirname "$0")/.."
mkdir -p data
rm -f data/gridnet_code.zip
zip -qr data/gridnet_code.zip gridnet scripts pyproject.toml -x '*__pycache__*' '*.pyc'
echo "wrote data/gridnet_code.zip ($(du -h data/gridnet_code.zip | cut -f1)) -> upload to Drive: GridNet/code/gridnet_code.zip"
