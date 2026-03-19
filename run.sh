#!/usr/bin/env bash
# run.sh — Autograder entrypoint for CS288 Assignment 3
# Usage: bash run.sh <questions_txt_path> <predictions_out_path>

set -e

# Ensure we run from submission root (paths are relative)
cd "$(dirname "$0")"

: "${RAG_ENABLE_RERANKER:=0}"
: "${RAG_RERANKER_BACKEND:=safe}"
: "${RAG_PROGRESS_LOGS:=1}"
: "${RAG_PROFILE_LLM:=1}"
: "${RAG_LLM_RETRIES:=1}"
: "${RAG_LLM_TIMEOUT:=25}"
: "${RAG_LLM_RETRY_SLEEP:=0.5}"
: "${PYTHONUNBUFFERED:=1}"
: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"

export RAG_ENABLE_RERANKER
export RAG_RERANKER_BACKEND
export RAG_PROGRESS_LOGS
export RAG_PROFILE_LLM
export RAG_LLM_RETRIES
export RAG_LLM_TIMEOUT
export RAG_LLM_RETRY_SLEEP
export PYTHONUNBUFFERED
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

python -u scripts/evaluate_rag_model.py "$QUESTIONS_PATH" "$PREDICTIONS_PATH"
