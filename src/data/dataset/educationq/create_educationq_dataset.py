#!/usr/bin/env python3
"""
EducationQ Dataset Creator

This script creates the EducationQ Dataset in three versions:
1. educationq_full.json - Complete dataset (MMLU-Pro Stratified + GPQA Diamond)
2. educationq_mmlu_pro_stratified.json - MMLU-Pro Stratified subset
3. educationq_gpqa_diamond.json - GPQA Diamond subset

EducationQ Dataset: A high-quality and balanced teaching-oriented testbed 
for evaluating LLMs' teaching capabilities.
"""

import json
import csv
import os
from typing import List, Dict, Any

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MMLU_PRO_PATH = os.path.join(SCRIPT_DIR, "../mmlu-pro/mmlu_pro_stratified.json")
GPQA_PATH = os.path.join(SCRIPT_DIR, "../gpqa/dataset/gpqa_diamond.csv")
OUTPUT_DIR = SCRIPT_DIR


def load_mmlu_pro_stratified() -> List[Dict[str, Any]]:
    """Load and format MMLU-Pro Stratified dataset."""
    with open(MMLU_PRO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        formatted_item = {
            "id": f"mmlu_pro_{item['question_id']}",
            "source": "mmlu-pro-stratified",
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"],
            "answer_index": item["answer_index"],
            "category": item["category"],
            "difficulty": "undergraduate",  # MMLU-Pro is undergraduate level
            "original_source": item.get("src", ""),
        }
        formatted_data.append(formatted_item)
    
    return formatted_data


def load_gpqa_diamond() -> List[Dict[str, Any]]:
    """Load and format GPQA Diamond dataset."""
    formatted_data = []
    
    with open(GPQA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Extract question and answers
            question = row.get("Question", "").strip()
            if not question:
                continue
            
            correct_answer = row.get("Correct Answer", "").strip()
            incorrect_1 = row.get("Incorrect Answer 1", "").strip()
            incorrect_2 = row.get("Incorrect Answer 2", "").strip()
            incorrect_3 = row.get("Incorrect Answer 3", "").strip()
            
            # Build options list (correct answer is always first, then we'll shuffle for actual use)
            options = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
            options = [opt for opt in options if opt]  # Remove empty options
            
            # Get subdomain/category
            subdomain = row.get("Subdomain", "").strip()
            high_level_domain = row.get("High-level domain", "").strip().lower()
            
            # Map to standard category
            if "physics" in high_level_domain.lower():
                category = "physics"
            elif "chemistry" in high_level_domain.lower():
                category = "chemistry"
            elif "biology" in high_level_domain.lower():
                category = "biology"
            else:
                category = high_level_domain if high_level_domain else "science"
            
            formatted_item = {
                "id": f"gpqa_diamond_{idx + 1}",
                "source": "gpqa-diamond",
                "question": question,
                "options": options,
                "answer": "A",  # Correct answer is always the first option
                "answer_index": 0,
                "category": category,
                "difficulty": "graduate",  # GPQA is graduate level
                "subdomain": subdomain,
                "explanation": row.get("Explanation", "").strip(),
            }
            formatted_data.append(formatted_item)
    
    return formatted_data


def create_dataset_info(full_data: List, mmlu_data: List, gpqa_data: List) -> Dict:
    """Create dataset metadata."""
    
    # Count categories
    def count_categories(data):
        cats = {}
        for item in data:
            cat = item["category"]
            cats[cat] = cats.get(cat, 0) + 1
        return dict(sorted(cats.items()))
    
    return {
        "dataset_name": "EducationQ Dataset",
        "version": "1.0.0",
        "description": "A high-quality and balanced teaching-oriented testbed for evaluating LLMs' teaching capabilities through multi-agent educational scenarios.",
        "citation": "@inproceedings{shi-etal-2025-educationq, title=\"{E}ducation{Q}: Evaluating {LLM}s' Teaching Capabilities Through Multi-Agent Dialogue Framework\", author=\"Shi, Yao and Liang, Rongkeng and Xu, Yong\", booktitle=\"Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)\", year=\"2025\", publisher=\"Association for Computational Linguistics\"}",
        "homepage": "https://github.com/SunriserFuture/EducationQ",
        "license": "MIT",
        "splits": {
            "full": {
                "description": "Complete EducationQ Dataset (MMLU-Pro Stratified + GPQA Diamond)",
                "num_examples": len(full_data),
                "categories": count_categories(full_data),
            },
            "mmlu_pro_stratified": {
                "description": "MMLU-Pro Stratified subset - 13 disciplines with 100 questions each",
                "num_examples": len(mmlu_data),
                "categories": count_categories(mmlu_data),
                "source_dataset": "TIGER-Lab/MMLU-Pro",
            },
            "gpqa_diamond": {
                "description": "GPQA Diamond subset - Graduate-level science questions",
                "num_examples": len(gpqa_data),
                "categories": count_categories(gpqa_data),
                "source_dataset": "Idavidrein/gpqa",
            },
        },
        "features": {
            "id": "Unique identifier for each question",
            "source": "Source dataset (mmlu-pro-stratified or gpqa-diamond)",
            "question": "The question text",
            "options": "List of answer options",
            "answer": "Correct answer letter (A, B, C, ...)",
            "answer_index": "Index of correct answer (0-based)",
            "category": "Subject/discipline category",
            "difficulty": "Difficulty level (undergraduate or graduate)",
        },
    }


def main():
    print("=" * 60)
    print("EducationQ Dataset Creator")
    print("=" * 60)
    
    # Load datasets
    print("\n[1/4] Loading MMLU-Pro Stratified...")
    mmlu_data = load_mmlu_pro_stratified()
    print(f"  ✓ Loaded {len(mmlu_data)} questions")
    
    print("\n[2/4] Loading GPQA Diamond...")
    gpqa_data = load_gpqa_diamond()
    print(f"  ✓ Loaded {len(gpqa_data)} questions")
    
    # Create full dataset
    print("\n[3/4] Creating combined dataset...")
    full_data = mmlu_data + gpqa_data
    print(f"  ✓ Total: {len(full_data)} questions")
    
    # Save datasets
    print("\n[4/4] Saving datasets...")
    
    # Save MMLU-Pro Stratified
    mmlu_output = os.path.join(OUTPUT_DIR, "educationq_mmlu_pro_stratified.json")
    with open(mmlu_output, "w", encoding="utf-8") as f:
        json.dump(mmlu_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {mmlu_output}")
    
    # Save GPQA Diamond
    gpqa_output = os.path.join(OUTPUT_DIR, "educationq_gpqa_diamond.json")
    with open(gpqa_output, "w", encoding="utf-8") as f:
        json.dump(gpqa_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {gpqa_output}")
    
    # Save Full dataset
    full_output = os.path.join(OUTPUT_DIR, "educationq_full.json")
    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {full_output}")
    
    # Save dataset info
    info = create_dataset_info(full_data, mmlu_data, gpqa_data)
    info_output = os.path.join(OUTPUT_DIR, "dataset_info.json")
    with open(info_output, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {info_output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Creation Complete!")
    print("=" * 60)
    print(f"\nEducationQ Full:          {len(full_data):,} questions")
    print(f"  - MMLU-Pro Stratified:  {len(mmlu_data):,} questions (13 disciplines)")
    print(f"  - GPQA Diamond:         {len(gpqa_data):,} questions (3 disciplines)")
    print("\nCategory Distribution:")
    
    cats = {}
    for item in full_data:
        cat = item["category"]
        cats[cat] = cats.get(cat, 0) + 1
    
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

