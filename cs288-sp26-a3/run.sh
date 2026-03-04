#!/usr/bin/env bash
# run.sh — Autograder entrypoint for CS288 Assignment 3
# Usage: bash run.sh <questions_txt_path> <predictions_out_path>

set -e

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

python3 evaluate_rag_model.py "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
