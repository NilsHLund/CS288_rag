#!/usr/bin/env bash
# run.sh — Autograder entrypoint for CS288 Assignment 3
# Usage: bash run.sh <questions_txt_path> <predictions_out_path>

set -e

# Ensure we run from submission root (paths are relative)
cd "$(dirname "$0")"

# Gradescope / low-RAM autograders: ~3× fewer LLM calls, sequential predict (see scripts/rag.py).
# For best local accuracy: CS288_RAG_FAST=0 bash run.sh …
export CS288_RAG_FAST="${CS288_RAG_FAST:-1}"

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

python3 scripts/evaluate_rag_model.py "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
