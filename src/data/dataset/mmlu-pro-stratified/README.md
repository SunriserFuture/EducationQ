---
license: apache-2.0
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - education
  - teaching
  - llm-evaluation
  - multi-agent
  - pedagogy
  - benchmark
  - mmlu
  - mmlu-pro
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: mmlu_pro_stratified.json
---

# MMLU-Pro-Stratified: A High-Quality & Balanced Teaching-Oriented Testbed

<p align="center">
  <img src="figures/mmlu-pro-vs-mmlu-pro-stratifited.png" alt="MMLU-Pro vs MMLU-Pro Stratified" width="850"/>
</p>

## 🌟 Definition and Value

**MMLU-Pro-Stratified** is a meticulously curated subset of MMLU-Pro, specifically designed to serve as a **high-quality and balanced teaching-oriented testbed** for Large Language Models (LLMs). 

### Why "Teaching-Oriented"?
Unlike traditional benchmarks that focus on single-turn accuracy, a **teaching-oriented testbed** evaluates a model's pedagogical capabilities:
- **Concept Explanation**: Can the model break down complex graduate/undergraduate topics?
- **Socratic Guiding**: Is the model capable of guiding a student through a problem-solving process rather than just giving the answer?
- **Robustness across Difficulties**: Does the teaching strategy remain effective as the problem difficulty increases?

### The Value of "Balance" (Stratification)
As shown in the chord diagram above, the original MMLU-Pro (left) exhibits significant imbalances in both discipline distribution and difficulty levels. **MMLU-Pro-Stratified** (right) resolves this through **Stratified Sampling**:
1. **Discipline Balance**: Precisely 100 questions for each of the 13 core disciplines, preventing subject bias.
2. **Difficulty Stratification**: Each discipline is evenly distributed across 10 difficulty ranges (based on Top-10 LLM average accuracy).
3. **Pedagogical Matrix**: This 13×10 matrix ensures that any evaluation of "teaching capability" is statistically representative across the entire knowledge landscape.

## 📊 Dataset Statistics

#### 1. The Stratification Matrix (1,300 Questions)

| Discipline | [0-10%) | [10-20%) | [20-30%) | [30-40%) | [40-50%) | [50-60%) | [60-70%) | [70-80%) | [80-90%) | [90-100%] | Total |
|------------|---------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-------|
| **13 Subjects** | 10 items | 10 items | 10 items | 10 items | 10 items | 10 items | 10 items | 10 items | 10 items | 10 items | **1,300** |

> **Note**: Subjects include: Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, and Psychology.

#### 2. Difficulty Taxonomy

Difficulty is calculated using the **Top-10 Model Average Accuracy**, providing a reliable proxy for question hardness.

| Level | Symbol | Accuracy Range | Questions | Interpretation |
|-------|--------|----------------|-----------|----------------|
| 1 | `+++++` | [0%, 10%) | 130 | Hardest (most models fail) |
| 2 | `++++` | [10%, 20%) | 130 | Very Hard |
| 3 | `+++` | [20%, 30%) | 130 | Hard |
| 4 | `++` | [30%, 40%) | 130 | Moderately Hard |
| 5 | `+` | [40%, 50%) | 130 | Slightly Hard |
| 6 | `-` | [50%, 60%) | 130 | Slightly Easy |
| 7 | `--` | [60%, 70%) | 130 | Easy |
| 8 | `---` | [70%, 80%) | 130 | Very Easy |
| 9 | `----` | [80%, 90%) | 130 | Easier |
| 10 | `-----` | [90%, 100%] | 130 | Easiest (most models succeed) |

> **Note**: 
> - `+` symbols indicate **harder** questions (more plus = harder)
> - `-` symbols indicate **easier** questions (more minus = easier)
> - GPQA Diamond questions do not have difficulty labels (field is empty)

## 🛠️ Usage in EducationQ Framework

This dataset is the primary testbed for the **EducationQ Framework**. It allows researchers to evaluate LLM "Teachers" in a controlled environment where the difficulty and subject matter are known variables.

```bash
# Evaluate a teacher model on the stratified MMLU-Pro set
python src/run/main.py --config config_teacher0shot_mmlupro_stratified.yaml
```

## 📜 Citation

If you use this balanced testbed, please cite our **ACL 2025** paper:

```bibtex
@inproceedings{shi-etal-2025-educationq,
    title = "{E}ducation{Q}: Evaluating {LLM}s' Teaching Capabilities Through Multi-Agent Dialogue Framework",
    author = "Shi, Yao  and
      Liang, Rongkeng  and
      Xu, Yong",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1576/"
}
```

## Dataset Structure

### Data Instance

```json
{
  "id": "mmlu_pro_70",
  "source": "mmlu-pro-stratified",
  "question": "Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.",
  "options": ["Safe practices, Fear, Jealousy, Trivial", "Unsafe practices, Distress, Joy, Trivial", "..."],
  "answer": "I",
  "answer_index": 8,
  "category": "business",
  "difficulty": "-----"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `mmlu_pro_70`) |
| `source` | string | Source dataset (`mmlu-pro-stratified`) |
| `question` | string | The question text |
| `options` | list[string] | List of 9-10 answer options |
| `answer` | string | Correct answer letter (A, B, C, ...) |
| `answer_index` | int | Index of correct answer (0-based) |
| `category` | string | Subject/discipline category |
| `difficulty` | string | Difficulty level (`+`/`-` symbols) |
| `original_source` | string | Original source from MMLU-Pro |

## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

# Load MMLU-Pro-Stratified dataset
dataset = load_dataset("SunriserFuture/MMLU-Pro-Stratified")

# Access data
for example in dataset["train"]:
    print(example["question"])
    print(example["options"])
    print(example["answer"])
    print(example["difficulty"])

# Filter by difficulty
hard_questions = dataset["train"].filter(lambda x: x["difficulty"] in ["+++++", "++++", "+++"])
easy_questions = dataset["train"].filter(lambda x: x["difficulty"] in ["-----", "----", "---"])

# Filter by category
math_questions = dataset["train"].filter(lambda x: x["category"] == "math")
```

### Loading with Python (Direct JSON)

```python
import json
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="SunriserFuture/MMLU-Pro-Stratified",
    filename="mmlu_pro_stratified.json",
    repo_type="dataset"
)

with open(file_path, "r") as f:
    data = json.load(f)

print(f"Total questions: {len(data)}")
```

### Integration with EducationQ Framework

```python
# Use with EducationQ Framework for teaching evaluation
# See: https://github.com/SunriserFuture/EducationQ

python src/run/main.py --config config_teacher0shot_mmlupro_stratified.yaml
```

## Source Dataset

This dataset is constructed from:

- **MMLU-Pro** ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro))
  - Enhanced version of MMLU with 10 answer options per question
  - 12,032 questions across 14 subjects
  - Stratified sampling: 100 questions per discipline × 13 disciplines

## Citation

If you use this dataset, please cite our ACL 2025 paper:

```bibtex
@inproceedings{shi-etal-2025-educationq,
    title = "{E}ducation{Q}: Evaluating {LLM}s' Teaching Capabilities Through Multi-Agent Dialogue Framework",
    author = "Shi, Yao  and
      Liang, Rongkeng  and
      Xu, Yong",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1576/",
    doi = "10.18653/v1/2025.acl-long.1576",
    pages = "32799--32828",
    ISBN = "979-8-89176-251-0",
}
```

Also please cite the original MMLU-Pro dataset:

```bibtex
@article{wang2024mmlu,
  title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and others},
  journal={arXiv preprint arXiv:2406.01574},
  year={2024}
}
```

## License

This dataset is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), following the original MMLU-Pro license.

## Related Datasets

- **EducationQ** ([SunriserFuture/EducationQ](https://huggingface.co/datasets/SunriserFuture/EducationQ)) - Complete EducationQ dataset (MMLU-Pro-Stratified + GPQA Diamond)
- **MMLU-Pro** ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) - Original dataset

## Contact

For questions and support:
- Email: educationq@sunriser.org
- GitHub: [https://github.com/SunriserFuture/EducationQ](https://github.com/SunriserFuture/EducationQ)

