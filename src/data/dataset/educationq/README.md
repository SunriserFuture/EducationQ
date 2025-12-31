---
license: cc-by-4.0
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
size_categories:
  - 1K<n<10K
configs:
  - config_name: educationq_full
    data_files: educationq_full.json
    default: true
  - config_name: mmlu_pro_stratified
    data_files: mmlu_pro_stratified.json
  - config_name: gpqa_diamond
    data_files: gpqa_diamond.json
---

# EducationQ Dataset: A Balanced Teaching-Oriented Testbed

<p align="center">
  <img src="figures/mmlu-pro-vs-mmlu-pro-stratifited.png" alt="MMLU-Pro vs MMLU-Pro Stratified" width="850"/>
</p>

## 🌟 Definition and Value

The **EducationQ Dataset** is a high-quality, balanced, and **teaching-oriented testbed** designed to evaluate the pedagogical capabilities of Large Language Models (LLMs). As detailed in our **ACL 2025** paper, this dataset serves as the foundational benchmark for the EducationQ multi-agent dialogue framework.

### Why "High-Quality and Balanced"?
A "teaching-oriented" evaluation requires more than just correct answers; it requires a dataset that covers a vast knowledge landscape without subject or difficulty bias.
- **High-Quality**: We combine the expert-validated **GPQA Diamond** (graduate-level) with the robust **MMLU-Pro** (undergraduate-level), ensuring questions are both challenging and accurate.
- **Balanced (Stratified)**: Unlike the original MMLU-Pro, which is heavily skewed towards certain subjects and easier difficulty ranges, the EducationQ subset (**MMLU-Pro Stratified**) uses stratified sampling to create a uniform distribution across 13 disciplines and 10 difficulty levels (as shown in the chord diagram above).
- **Teaching-Oriented**: Each question is treated as a "teaching task," where a model's success is measured by its ability to explain concepts and guide students through multi-turn interactions.

### The EducationQ Matrix
By stratifying data into a **13 subjects × 10 difficulty levels** matrix, we provide a "calibrated scale" for teaching ability. This allows researchers to pinpoint exactly where a teacher model fails—whether it's on graduate-level physics or undergraduate-level law.

## 📊 Dataset Summary

#### 1. Dataset Statistics

| Version | Questions | Disciplines |
|---------|-----------|-------------|
| **EducationQ Full** | 1,498 | 13 |
| **MMLU-Pro Stratified** | 1,300 | 13 |
| **GPQA Diamond** | 198 | 3 |


#### 2. Difficulty Levels (10 levels for MMLU-Pro Stratified)

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

#### 3. Disciplines

| # | Discipline | Count (MMLU-Pro Stratified) | Count (GPQA Diamond) |
|---|------------|------------------|--------------|
| 1 | Biology | 100 | 19 |
| 2 | Business | 100 | - |
| 3 | Chemistry | 100 | 93 |
| 4 | Computer Science | 100 | - |
| 5 | Economics | 100 | - |
| 6 | Engineering | 100 | - |
| 7 | Health | 100 | - |
| 8 | History | 100 | - |
| 9 | Law | 100 | - |
| 10 | Math | 100 | - |
| 11 | Philosophy | 100 | - |
| 12 | Physics | 100 | 86 |
| 13 | Psychology | 100 | - |


### Supported Tasks

- **LLM Teaching Capability Evaluation**: Assess how effectively LLMs can teach students through multi-turn interactions
- **Educational Agent Benchmarking**: Compare teaching strategies and pedagogical effectiveness across different LLMs
- **Multi-Agent Educational Scenarios**: Simulate dynamic teacher-student interactions

## Dataset Structure

### Data Instances

**MMLU-Pro Stratified Example:**
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

**GPQA Diamond Example:**
```json
{
  "id": "gpqa_diamond_1",
  "source": "gpqa-diamond",
  "question": "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec...",
  "options": ["10^-4 ev", "10^-11 ev", "10^-8 ev", "10^-9 ev"],
  "answer": "D",
  "answer_index": 3,
  "category": "physics",
  "difficulty": "",
  "subdomain": "Physics (general)"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `mmlu_pro_70`, `gpqa_diamond_1`) |
| `source` | string | Source dataset (`mmlu-pro-stratified` or `gpqa-diamond`) |
| `question` | string | The question text |
| `options` | list[string] | List of answer options (9-10 for MMLU-Pro, 4 for GPQA) |
| `answer` | string | Correct answer letter (A, B, C, ...) |
| `answer_index` | int | Index of correct answer (0-based) |
| `category` | string | Subject/discipline category |
| `difficulty` | string | Difficulty level (`+`/`-` symbols for MMLU-Pro, empty for GPQA) |
| `subdomain` | string | (GPQA only) Specific subdomain |
| `explanation` | string | (GPQA only) Answer explanation |

### Dataset Subsets

| Subset | File | Questions | Description |
|--------|------|-----------|-------------|
| `educationq_full` (default) | `educationq_full.json` | 1,498 | Complete dataset |
| `mmlu_pro_stratified` | `mmlu_pro_stratified.json` | 1,300 | MMLU-Pro subset |
| `gpqa_diamond` | `gpqa_diamond.json` | 198 | GPQA Diamond subset |

## Dataset Distribution

<p align="center">
  <img src="figures/dataset-distribution.png" alt="Dataset Distribution" width="600"/>
</p>

### MMLU-Pro Stratified (1,300 questions)

Stratified sampling from [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) with **100 questions per discipline**:

| # | Discipline | Count | Percentage |
|---|------------|-------|------------|
| 1 | Business | 100 | 6.68% |
| 2 | Law | 100 | 6.68% |
| 3 | Psychology | 100 | 6.68% |
| 4 | Biology | 100 | 6.68% |
| 5 | Chemistry | 100 | 6.68% |
| 6 | History | 100 | 6.68% |
| 7 | Health | 100 | 6.68% |
| 8 | Economics | 100 | 6.68% |
| 9 | Math | 100 | 6.68% |
| 10 | Physics | 100 | 6.68% |
| 11 | Engineering | 100 | 6.68% |
| 12 | Philosophy | 100 | 6.68% |
| 13 | Computer Science | 100 | 6.68% |

### Difficulty Levels (MMLU-Pro Stratified)

The `difficulty` field for MMLU-Pro Stratified is based on the average accuracy of top-10 LLMs on each question:

| Accuracy Range | Difficulty | Interpretation |
|---------------|------------|----------------|
| [0%, 10%) | `+++++` | Hardest (most models fail) |
| [10%, 20%) | `++++` | Very Hard |
| [20%, 30%) | `+++` | Hard |
| [30%, 40%) | `++` | Moderately Hard |
| [40%, 50%) | `+` | Slightly Hard |
| [50%, 60%) | `-` | Slightly Easy |
| [60%, 70%) | `--` | Easy |
| [70%, 80%) | `---` | Very Easy |
| [80%, 90%) | `----` | Easier |
| [90%, 100%] | `-----` | Easiest (most models succeed) |

**Interpretation**:
- `+` symbols indicate **harder** questions (more plus = harder)
- `-` symbols indicate **easier** questions (more minus = easier)

**Note**: GPQA Diamond questions do not have difficulty labels (field is empty).

### GPQA Diamond (198 questions)

Graduate-level science questions from [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa):

| # | Discipline | Count | Percentage |
|---|------------|-------|------------|
| 14 | Physics | 86 | 5.74% |
| 15 | Chemistry | 93 | 6.21% |
| 16 | Biology | 19 | 1.27% |

## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

# Load complete EducationQ dataset (default, 1,498 questions)
dataset = load_dataset("SunriserFuture/EducationQ")
# or explicitly: load_dataset("SunriserFuture/EducationQ", "educationq_full")

# Load MMLU-Pro Stratified subset (1,300 questions)
mmlu_dataset = load_dataset("SunriserFuture/EducationQ", "mmlu_pro_stratified")

# Load GPQA Diamond subset (198 questions)
gpqa_dataset = load_dataset("SunriserFuture/EducationQ", "gpqa_diamond")

# Access data
for example in dataset["train"]:
    print(example["question"])
    print(example["options"])
    print(example["answer"])
```

### Available Subsets

| Subset | Description | Questions |
|--------|-------------|-----------|
| `educationq_full` (default) | Complete EducationQ dataset | 1,498 |
| `mmlu_pro_stratified` | MMLU-Pro Stratified subset | 1,300 |
| `gpqa_diamond` | GPQA Diamond subset | 198 |

### Loading with Python (Direct JSON)

```python
import json
from huggingface_hub import hf_hub_download

# Download and load specific subset
file_path = hf_hub_download(
    repo_id="SunriserFuture/EducationQ",
    filename="educationq_full.json",  # or mmlu_pro_stratified.json, gpqa_diamond.json
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

## Source Datasets

This dataset is constructed from:

1. **MMLU-Pro** ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro))
   - Enhanced version of MMLU with 10 answer options per question
   - 12,032 questions across 14 subjects
   - Stratified sampling: 100 questions per discipline

2. **GPQA** ([Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa))
   - Graduate-level science questions
   - Diamond subset: 198 expert-validated questions
   - Disciplines: Physics, Chemistry, Biology

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

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

### Source Dataset Licenses

| Dataset | License | Attribution |
|---------|---------|-------------|
| **MMLU-Pro** | Apache 2.0 | TIGER-Lab ([GitHub](https://github.com/TIGER-AI-Lab/MMLU-Pro), [HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) |
| **GPQA** | CC BY 4.0 | David Rein et al. ([GitHub](https://github.com/idavidrein/gpqa), [HuggingFace](https://huggingface.co/datasets/Idavidrein/gpqa)) |

### Attribution Requirements

When using this dataset, please:
1. Cite the EducationQ paper (see Citation section above)
2. Acknowledge the source datasets:
   - MMLU-Pro: Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark"
   - GPQA: Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"

## Contact

For questions and support:
- Email: educationq@sunriser.org
- GitHub: [https://github.com/SunriserFuture/EducationQ](https://github.com/SunriserFuture/EducationQ)

