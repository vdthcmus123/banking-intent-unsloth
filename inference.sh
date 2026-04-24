#!/bin/bash
set -e

echo "=== Banking Intent Detection — Inference ==="

python scripts/inference.py configs/inference.yaml sample_data/test.csv

echo "Inference hoàn tất!"
