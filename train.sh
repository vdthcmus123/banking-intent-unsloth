#!/bin/bash
set -e

echo "=== Banking Intent Detection — Training Pipeline ==="

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo "[2/3] Preprocessing data..."
python scripts/preprocess_data.py

echo "[3/3] Training model..."
python scripts/train.py

echo "Training hoàn tất!"
