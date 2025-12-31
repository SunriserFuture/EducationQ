"""EducationQ Dataset loading script for HuggingFace datasets library."""

import json
import os
from typing import List

import datasets


_CITATION = """\
@inproceedings{shi-etal-2025-educationq,
    title = "{E}ducation{Q}: Evaluating {LLM}s' Teaching Capabilities Through Multi-Agent Dialogue Framework",
    author = "Shi, Yao  and
      Liang, Rongkeng  and
      Xu, Yong",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1576/",
    doi = "10.18653/v1/2025.acl-long.1576",
    pages = "32799--32828",
}
"""

_DESCRIPTION = """\
EducationQ Dataset is a high-quality and balanced teaching-oriented testbed 
for evaluating LLMs' teaching capabilities through multi-agent educational scenarios.
"""

_HOMEPAGE = "https://github.com/SunriserFuture/EducationQ"

_LICENSE = "CC BY 4.0"

_URLS = {
    "educationq-full": "data/educationq-full/train.json",
    "mmlu-pro-stratified": "data/mmlu-pro-stratified/train.json",
    "gpqa-diamond": "data/gpqa-diamond/train.json",
}


class EducationQConfig(datasets.BuilderConfig):
    """BuilderConfig for EducationQ."""

    def __init__(self, **kwargs):
        super(EducationQConfig, self).__init__(**kwargs)


class EducationQ(datasets.GeneratorBasedBuilder):
    """EducationQ Dataset: A teaching-oriented testbed for LLMs."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        EducationQConfig(
            name="educationq-full",
            version=VERSION,
            description="Complete EducationQ Dataset (MMLU-Pro Stratified + GPQA Diamond) - 1,498 questions",
        ),
        EducationQConfig(
            name="mmlu-pro-stratified",
            version=VERSION,
            description="MMLU-Pro Stratified subset - 1,300 questions across 13 disciplines",
        ),
        EducationQConfig(
            name="gpqa-diamond",
            version=VERSION,
            description="GPQA Diamond subset - 198 graduate-level science questions",
        ),
    ]

    DEFAULT_CONFIG_NAME = "educationq-full"

    def _info(self):
        # Note: difficulty is int (1-10) for MMLU-Pro, empty string for GPQA
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "source": datasets.Value("string"),
                "question": datasets.Value("string"),
                "options": datasets.Sequence(datasets.Value("string")),
                "answer": datasets.Value("string"),
                "answer_index": datasets.Value("int32"),
                "category": datasets.Value("string"),
                "difficulty": datasets.Value("string"),  # "1"-"10" or "" (empty)
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_path = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            # Convert difficulty to string (int 1-10 for MMLU-Pro, "" for GPQA)
            difficulty = item.get("difficulty", "")
            if isinstance(difficulty, int):
                difficulty = str(difficulty)
            elif difficulty is None:
                difficulty = ""
            
            yield idx, {
                "id": item.get("id", str(idx)),
                "source": item.get("source", ""),
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "answer": item.get("answer", ""),
                "answer_index": item.get("answer_index", 0),
                "category": item.get("category", ""),
                "difficulty": difficulty,
            }

