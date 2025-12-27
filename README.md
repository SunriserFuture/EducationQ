# EducationQ Framework

<p align="center">
  <img src="docs/figures/interaction-process.png" alt="Interaction Process" width="800"/>
</p>

**EducationQ Framework** is a comprehensive multi-agent educational framework that transforms and evaluates LLMs' teaching capabilities through simulated dynamic educational scenarios. Grounded in pedagogical theories (Zone of Proximal Development, Scaffolding, Informal Formative Assessment, Bloom's Taxonomy), EducationQ enables LLMs to consolidate their foundational abilities—knowledge retrieval, generation, and reasoning—into comprehensive teaching effectiveness.

## 📊 EducationQ Benchmark

<p align="center">
  <img src="docs/figures/results-1.png" alt="Overall Results" width="800"/>
</p>

**EducationQ Benchmark** is a quantitative and qualitative mixed-method evaluation of LLMs' teaching capabilities based on the EducationQ Framework. 

Constructing research of 14 LLMs across major AI Organizations(OpenAI, Meta, Google, Anthropic, etc.) through mixed-methods approach, we **observe LLMs' distinct teaching behaviors & strategies** and **benchmark capabilities via student learning gains**, revealing the need for pedagogical optimization beyond knowledge scaling.

<p align="center">
  <img src="docs/figures/results-2.png" alt="Per-Subject Results" width="800"/>
</p>

## 🎯 Key Insights

Our research reveals several important findings:

- **Teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities** - some smaller open-source models outperform larger commercial counterparts in teaching contexts
- **78% agreement** between human expert evaluations and our automated qualitative analysis of effective teaching behaviors
- **LLMs-as-Teachers require specialized optimization** beyond simple scaling, suggesting next-generation educational AI should prioritize targeted enhancement of specific pedagogical effectiveness

## 📖 Citation

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

## 📚 Interdisciplinary Grounding

EducationQ is grounded in established educational theories:

1. **Zone of Proximal Development (ZPD)** *(Vygotsky, 1978)* — Defines the ideal learning space where effective teaching must occur
2. **Scaffolding** *(Wood, Bruner, & Ross, 1976)* — Providing temporary, adaptive support—giving hints, not answers
3. **Strategic Questioning & Bloom's Taxonomy** *(Bloom, 1956)* — Provides a measurable framework to assess teaching quality
4. **Informal Formative Assessment (IFA)** *(Ruiz-Primo & Furtak, 2007)* — The engine that drives dynamic, personalized teaching

## 🔬 Multi-Agent Framework

<p align="center">
  <img src="docs/figures/dynamic-educational-scenario-2.png" alt="Dynamic Educational Scenario" width="800"/>
</p>

EducationQ employs three specialized agents:

- **Student Agent**: Takes tests, reflects on feedback and questions, answers with understanding
- **Teacher Agent**: Evaluates responses, adjusts strategy, provides feedback, teaches by questioning
- **Evaluator Agent**: Calculates accuracy, verifies compliance, analyzes effectiveness, measures learning gains

### Evaluation Pipeline

The framework operates through three phases:

1. **Pre-Test**: Student takes initial test; Evaluator calculates baseline accuracy
2. **Interaction**: Multi-turn dialogue where Teacher guides Student through questioning and feedback
3. **Post-Test**: Student retakes test; Evaluator measures learning gains

## 📁 Datasets

<p align="center">
  <img src="docs/figures/mmlu-pro-vs-mmlu-pro-stratifited.png" alt="MMLU-Pro vs MMLU-Pro Stratified" width="600"/>
</p>

EducationQ supports multiple benchmark datasets and user-defined datasets:

| Dataset | Questions | Domains | Description |
|---------|-----------|---------|-------------|
| **MMLU-Pro** | 12,032 | 14 subjects | Enhanced MMLU with 10 options per question |
| **GPQA** | 448 | Science | Graduate-level science questions |
| **AGIEval** | Varies | Multiple | Human-centric benchmark tasks |

And we construct **MMLU-Pro Stratified + GPQA Diamond —— a high-quality and balanced teaching-oriented testbed** for LLMs' teaching capabilities evaluation.

### MMLU-Pro Stratified + GPQA Diamond

<p align="center">
  <img src="docs/figures/dataset-distribution.png" alt="Dataset Distribution" width="400"/>
</p>

1,498 Teaching Tasks covers **13 different disciplines** and **10 difficulty levels**: Meticulously curated from two elite benchmark data sources.

Pre-filtered stratified subsets are available:
- `mmlu_pro_stratified.json`: 1,300 stratified questions from MMLU-Pro
- `gpqa_diamond.csv`: Diamond subset of GPQA


Our Goal: To construct a robust evaluation dataset that moves beyond simple knowledge recall to assess deep and  pedagogical interaction, isolating and measuring genuine pedagogical skills.


## 📊 Evaluation Metrics

<p align="center">
  <img src="docs/figures/evaluating-metrics.png" alt="Evaluation Metrics" width="800"/>
</p>

EducationQ provides both **quantitative** and **qualitative** evaluation:

### Quantitative Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **ALG** (Absolute Learning Gain) | Direct improvement in student performance | ACC_post - ACC_pre |
| **PNIR** (Positive-Negative Impact Ratio) | Consistency of teaching effectiveness | N_neg / N_pos |
| **CSS** (Cross-Subject Stability) | Standard deviation of learning gains across subjects | σ(SLGPD) |
| **UIC** (Unique Improvement Count) | Questions where only one specific teacher model achieved improvement | Count(OUI) |

### Qualitative Analysis Dimensions

**Holistic Interaction Analysis:**
1. Assessment Effectiveness
2. Questioning Effectiveness
3. Feedback Effectiveness
4. Instructional Adaptation Effectiveness
5. Learning Objective Achievement Effectiveness

**Teacher-Centric Question Analysis:**
1. Question Relevance
2. Cognitive Level
3. Knowledge Dimension
4. Question Diversity
5. Scaffolding Progression
6. Metacognitive Promotion

**Student-Centric Response Analysis:**
1. Response Relevance
2. Cognitive Level Demonstration
3. Knowledge Dimension Integration
4. Response Diversity
5. Elaboration Progression
6. Metacognitive Reflection

## ✨ Features

- **Multi-Agent Architecture**: Teacher and Student agents powered by different LLMs
- **Multiple Datasets**: Support for MMLU-Pro, GPQA, and AGIEval datasets  
- **Flexible Evaluation**: Both quantitative (accuracy-based) and qualitative (interaction-based) analysis
- **Resume Capability**: Can continue from any stage using saved results
- **Comprehensive Analysis**: Multiple evaluation perspectives (interaction, teacher questions, student responses)
- **Local Dataset Support**: Load datasets from local JSON files without network dependency

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/SunriserFuture/EducationQ.git
cd EducationQ/EducationQ_Framework

# Install dependencies
pip install -r requirements.txt
```

## 📋 Quick Start

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

## ⚙️ Configuration

The framework uses YAML configuration files. See `src/data/input/config_template.yaml` for a complete example.

### Key Configuration Sections

#### Dataset Configuration
```yaml
DATASET_TYPE: "mmlu-pro"  # Options: "gpqa", "mmlu-pro", "agieval"
DATASET_NAME: "TIGER-Lab/MMLU-Pro"  # Or local file: "mmlu_pro_stratified.json"
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

## 🖥️ Command Line Options

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

## 📤 Output Files

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

## 📂 Directory Structure

```
EducationQ_Framework/
├── docs/
│   ├── figures/                    # Framework diagrams and result visualizations
│   └── 2025.acl-long.1576.pdf     # ACL 2025 paper
├── src/
│   ├── data/
│   │   ├── input/                  # Configuration files
│   │   │   ├── config_template.yaml
│   │   │   ├── config_teacher0shot_gpqa_diamond.yaml
│   │   │   └── config_teacher0shot_mmlupro_stratified.yaml
│   │   ├── dataset/                # Dataset files
│   │   │   ├── gpqa/              # GPQA dataset
│   │   │   ├── AGIEval/           # AGIEval dataset
│   │   │   └── mmlu-pro/          # MMLU-Pro dataset
│   │   └── output/                 # Experiment results
│   └── run/
│       └── main.py                 # Main entry point
├── README.md
└── requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [docs/contributing.md](docs/contributing.md) for detailed guidelines.

## 📄 License

MIT License

## 📧 Contact

For questions and support, please contact: educationq@sunriser.org

---

<p align="center">
  <b>EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework</b><br>
  ACL 2025 | <a href="https://aclanthology.org/2025.acl-long.1576/">Paper</a> | <a href="https://github.com/SunriserFuture/EducationQ">Code</a>
</p>
