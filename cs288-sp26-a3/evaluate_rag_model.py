#!/usr/bin/env python3

"""
Evaluate a RAG QA model.
Usage:
 python evaluate_rag_model.py data/questions.txt data/answers.txt
Assumptions:
- from rag import RAGModel works.
- model.predict(List[str]) -> List[str]
- Input file: one question per line.
- Output file: one answer per line (aligned).
"""

# A hack to make environment variables work for me, it is not automatically working for some reason
import os
with open(".env") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ[key] = value
            
import argparse
import time
from pathlib import Path
from typing import List
from rag import RAGModel

def read_questions(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def write_answers(path: Path, answers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ans in answers:
            ans = (ans or "").replace("\n", " ").strip()
            f.write(ans + "\n")
            
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG QA model.")
    parser.add_argument("input_path", type=str, help="Path to questions.txt")
    parser.add_argument("output_path", type=str, help="Path to save answers.txt")
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # Step 1: Load QA model
    model = RAGModel()
    
    # Step 2: Load questions
    questions = read_questions(input_path)
    
    # Step 3: Get predictions (timed)
    start_time = time.perf_counter()
    answers = model.predict(questions)
    end_time = time.perf_counter()
    predict_time = end_time - start_time
    print(f"Step 3 (model prediction) took {predict_time:.4f} seconds.")
    
    if len(answers) != len(questions):
        raise RuntimeError(
        f"Number of answers ({len(answers)}) "
        f"does not match number of questions ({len(questions)})."
        )
        
    # Step 4: Save predictions
    write_answers(output_path, answers)
    print(f"Saved {len(answers)} answers to {output_path}")
    

if __name__ == "__main__":
    main()