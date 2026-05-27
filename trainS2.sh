#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=0 python3 FSP/train.py -opt options/train_FSPS2.yml --launcher none
