# EducationQ Framework

A comprehensive multi-agent educational evaluation system for assessing teaching effectiveness through AI-powered teacher-student interactions.

## Citation

If you use this framework in your research, please cite our paper:

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
    pages = "32799--32828",
    ISBN = "979-8-89176-251-0"
}
```

## Overview

EducationQ Framework is designed to evaluate the effectiveness of different teaching approaches by simulating teacher-student interactions using large language models. The framework supports multiple datasets (MMLU-Pro, GPQA, AGIEval) and provides both quantitative and qualitative analysis of teaching performance.

This work is based on the research presented in our ACL 2025 paper (2025.acl-long.1576.pdf), which introduces a novel multi-agent framework for evaluating large language models' teaching capabilities through simulated educational interactions.

## Features

- **Multi-Agent Architecture**: Teacher and Student agents powered by different LLMs
- **Multiple Datasets**: Support for MMLU-Pro, GPQA, and AGIEval datasets  
- **Flexible Evaluation**: Both quantitative (accuracy-based) and qualitative (interaction-based) analysis
- **Resume Capability**: Can continue from any stage using saved results
- **Comprehensive Analysis**: Multiple evaluation perspectives (interaction, teacher questions, student responses)

## Installation

```bash
# Clone the repository
git clone https://github.com/SunriserFuture/EducationQ.git
cd EducationQ/EducationQ_Framework

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

Run the complete evaluation pipeline:

```bash
python src/run/main.py
```

### 2. Custom Configuration

Use a custom configuration file:

```bash
python src/run/main.py --config ../data/input/my_config.yaml
```

### 3. Resume from Previous Results

Load existing pretest results and continue:

```bash
python src/run/main.py --mode load_pretest --input pretest_results.json
```

Load existing interaction results and continue:

```bash
python src/run/main.py --mode load_interaction --input interaction_results.json
```

### 4. Run Specialized Evaluations

Run comprehensive evaluation on existing results:

```bash
python src/run/main.py --mode evaluation --posttest posttest.json --csv evaluation_tasks.csv --eval-type comprehensive
```

## Configuration

The framework uses YAML configuration files. See `src/data/input/config_template.yaml` for a complete example.

### Key Configuration Sections

#### Dataset Configuration
```yaml
DATASET_TYPE: "mmlu-pro"  # Options: "gpqa", "mmlu-pro", "agieval"
DATASET_NAME: "TIGER-Lab/MMLU-Pro"
SELECTED_CATEGORIES: []  # Empty for all categories
SELECTED_QUESTION_ID: []  # Empty for all questions
```

#### Teacher Configuration
```yaml
TEACHER_CONFIGS:
  - name: "Teacher1"
    model: "meta-llama/llama-3.1-70b-instruct"
    api_key: "your_api_key"
    base_url: "your_api_url"
    temperature: 0.0
    max_tokens: 1024
    use_few_shot: false
    recommended_question_token_limit: 150
```

#### Student Configuration
```yaml
STUDENT_CONFIGS:
  - name: "Student1"
    model: "meta-llama/llama-3.1-70b-instruct"
    api_key: "your_api_key"
    base_url: "your_api_url"
    temperature: 0.0
    answer_max_tokens: 1024
    test_max_tokens: 2048
    include_pretest_info: true
```

#### Evaluator Configuration
```yaml
EVALUATOR_CONFIG:
  name: "Evaluator"
  model: "openai/gpt-4o-mini"
  api_key: "your_openai_api_key"
  base_url: "your_api_url"
  temperature: 0.0
  max_tokens: 4096
```

## Evaluation Types

### 1. Default Evaluation (Quantitative)

**Function**: `manager._perform_evaluation(posttest_results)`

**Purpose**: Quantitative analysis of student performance improvement

**Input**: Posttest results from the pipeline

**Output**: 
- Pre-test vs post-test accuracy comparison
- Progress metrics by category and overall
- Student and teacher configuration details

**Example Output**:
```json
{
  "Teacher1": {
    "Student1": {
      "overall": {
        "pre_test_accuracy": 0.65,
        "post_test_accuracy": 0.78,
        "progress": 0.13
      },
      "category_name": {
        "pre_test_accuracy": 0.60,
        "post_test_accuracy": 0.75,
        "progress": 0.15
      }
    }
  }
}
```

### 2. Specialized Evaluations (Qualitative)

These evaluations require a CSV file specifying which teacher pairs to compare for which questions.

**CSV Format**:
```csv
question_id,teacher_a,teacher_b
10822,Teacher1,Teacher2
10935,Teacher1,Teacher3
```

#### a) Interaction Evaluation

**Function**: `manager._interaction_evaluation(posttest_results_path, csv_path)`

**Purpose**: Analyzes the entire teacher-student conversation process

**Evaluation Dimensions**:
- Assessment Effectiveness
- Questioning Effectiveness  
- Feedback Effectiveness
- Instructional Adaptation Effectiveness
- Learning Objective Achievement Effectiveness

#### b) Teacher Questions Evaluation

**Function**: `manager._teacher_questions_evaluation(posttest_results_path, csv_path)`

**Purpose**: Focuses only on teacher-generated questions

**Evaluation Dimensions**:
- Question Relevance
- Cognitive Level
- Knowledge Dimension
- Question Diversity
- Scaffolding Progression
- Metacognitive Promotion

#### c) Student Responses Evaluation

**Function**: `manager._student_responses_evaluation(posttest_results_path, csv_path)`

**Purpose**: Focuses only on student-generated responses

**Evaluation Dimensions**:
- Response Relevance
- Cognitive Level Demonstration
- Knowledge Dimension Integration
- Response Diversity
- Elaboration Progression
- Metacognitive Reflection

#### d) Comprehensive Evaluation

**Function**: `manager._comprehensive_evaluation(posttest_results_path, csv_path)`

**Purpose**: Combines all three specialized analyses

**Output**: All three evaluation types combined in one result

## Command Line Options

```bash
python src/run/main.py [OPTIONS]
```

**Options**:
- `--config PATH`: Configuration YAML file path (default: `../data/input/config_template.yaml`)
- `--mode MODE`: Execution mode:
  - `complete`: Full pipeline (default)
  - `load_pretest`: Load pretest results and continue
  - `load_interaction`: Load interaction results and continue
  - `evaluation`: Run specific evaluation on existing results
- `--input PATH`: Input JSON file for `load_pretest` or `load_interaction` modes
- `--posttest PATH`: Posttest results JSON file for evaluation mode
- `--csv PATH`: CSV file with evaluation tasks for evaluation mode
- `--eval-type TYPE`: Evaluation type for evaluation mode:
  - `interaction`: Analyze conversation process
  - `teacher_questions`: Analyze teacher questions only
  - `student_responses`: Analyze student responses only
  - `comprehensive`: All three analyses (default)

## Output Files

The framework generates several types of output files:

1. **Pretest Results**: `pretest_results_{version}_{timestamp}.json`
2. **Interaction Results**: `pretest_interaction_results_{version}_{timestamp}.json`
3. **Posttest Results**: `pretest_interaction_posttest_results_{version}_{timestamp}.json`
4. **Evaluation Results**: `evaluation_results_{version}_{timestamp}.json`
5. **Specialized Evaluations**: 
   - `interaction_evaluation_results_{version}_{timestamp}.json`
   - `teacher_questions_evaluation_results_{version}_{timestamp}.json`
   - `student_responses_evaluation_results_{version}_{timestamp}.json`
   - `comprehensive_evaluation_results_{version}_{timestamp}.json`

## Directory Structure

```
EducationQ_Framework/
├── src/
│   ├── data/
│   │   ├── input/
│   │   │   └── config_template.yaml
│   │   ├── dataset/
│   │   │   ├── gpqa/
│   │   │   ├── AGIEval/
│   │   │   └── mmlu-pro/
│   │   └── output/
│   └── run/
│       └── main.py
├── README.md
└── requirements.txt
```

## Supported Models

The framework supports various LLM providers:

- **OpenAI**: GPT-4, GPT-4o-mini, etc.
- **Meta**: Llama models
- **Google**: Vertex AI models
- **Custom APIs**: Any OpenAI-compatible API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License

## Contact

For questions and support, please contact: educationq@sunriser.org 