import os
import json
import csv
import re
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from collections import defaultdict
import time
from functools import wraps
import pandas as pd
import tiktoken
import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import logging
from logging.handlers import RotatingFileHandler
import ast
import google.auth
import google.auth.transport.requests
from google.auth import impersonated_credentials

from pydantic import BaseModel

# Defined setting
CONFIG = {
    "API_CALL_MAX_RETRIES": 5,
    "API_CALL_INITIAL_DELAY": 10,
    "API_CALL_MAX_DELAY": 320,
    "PARALLEL_TASKS": 5,
    "CONFIG_YAML_FILEPATH": "D:/Workspace/EducationQ_Benchmark/src/data/input/config_teacher0shot_gpqa_diamond_0928_student_ablation.yaml",
}

# Configuration
class EvalConfig:
    def __init__(self, config_data: Dict[str, Any]):
        # system_setting
        self.output_path = config_data.get("OUTPUT_PATH", "./output")
        self.logging_level = config_data.get("LOGGING_LEVEL", "INFO")

        # eval_process_setting
        self.experiment_version = config_data.get("EXPERIMENT_VERSION", "1.0.0")
        self.num_interactions = config_data.get("NUM_INTERACTIONS", 5)
        self.num_if_few_shots = config_data.get("NUM_IF_FEW_SHOTS", 5)

        # dataset_setting
        ## "DATASET_TYPE": "gpqa", "mmlu-pro", "agieval"
        ## "DATASET_NAME": "gpqa_diamond.csv", "gpqa_experts.csv", "gpqa_extended.csv", "gpqa_main.csv", "TIGER-Lab/MMLU-Pro", "aqua-rat.jsonl", "sat-math.jsonl"
        self.dataset_type = config_data.get("DATASET_TYPE", "mmlu-pro")
        self.dataset_name = config_data.get("DATASET_NAME", "TIGER-Lab/MMLU-Pro")
        self.selected_categories = config_data.get("SELECTED_CATEGORIES", [])
        self.selected_question_ids = config_data.get("SELECTED_QUESTION_ID", [])
        self.first_questions_size = config_data.get("FIRST_QUESTIONS_SIZE")
        self.questions_sample_size = config_data.get("QUESTIONS_SAMPLE_SIZE")
        self.gpqa_test_data_folder_path = config_data.get("GPQA_TEST_DATA_FOLDER_PATH", "./src/data/dataset/gpqa/dataset/")
        self.gpqa_val_data_filepath = config_data.get("GPQA_VAL_DATA_FILEPATH", "./src/data/dataset/gpqa/prompts/chain_of_thought_examples.json")
        self.agieval_test_data_folder_path = config_data.get("AGIEVAL_TEST_DATA_FOLDER_PATH", "D:/Workspace/EducationQ_Benchmark/src/data/dataset/AGIEval/data/v1_1/")
        self.agieval_val_data_filepath = config_data.get("AGIEVAL_VAL_DATA_FILEPATH", "D:/Workspace/EducationQ_Benchmark/src/data/dataset/AGIEval/data/few_shot_prompts.csv")
        self.agieval_dataset_names = config_data.get("AGIEVAL_DATASET_NAMES", ["aqua-rat", "sat-math"])

        # teachers_setting
        self.teacher_configs = config_data.get("TEACHER_CONFIGS", [])

        # students_setting
        self.student_configs = config_data.get("STUDENT_CONFIGS", [])

        # evaluator_setting
        self.evaluator_config = config_data.get("EVALUATOR_CONFIG", {})

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            #update_config(config_data)  
        return cls(config_data)
    

# Utility functions
def setup_logging(logging_level, output_path):
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    logging_level = logging_level.upper()
    if logging_level not in valid_levels:
        raise ValueError(f'Invalid log level: {logging_level}. Valid levels are: {", ".join(valid_levels)}')
    
    numeric_level = getattr(logging, logging_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    log_file = os.path.join(output_path, 'educationq_benchmark_info.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=20*1024*1024, backupCount=10)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    log_file_1 = os.path.join(output_path, 'educationq_benchmark_warning.log')
    warning_handler = RotatingFileHandler(log_file_1, maxBytes=20*1024*1024, backupCount=10)
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    root_logger.addHandler(warning_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")


def retry_api_call(max_retries, initial_delay, max_delay=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(
                        f"Error in API call (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}"
                    )
                    if attempt == max_retries - 1:
                        logging.error("Max retries reached. Returning None.")
                        return None
                    logging.info(f"Waiting for {delay} seconds before next retry.")
                    time.sleep(delay)
                    delay *= 2
                    if max_delay and delay > max_delay:
                        delay = max_delay
        return wrapper
    return decorator

def save_progress(progress, filename):
    with open(filename, "w") as f:
        json.dump(progress, f)

def load_progress(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}




# Base LLM class
class BaseLLM:
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_few_shot: bool = False,
        num_if_few_shots: int = 5,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_few_shot = use_few_shot
        self.num_if_few_shots = num_if_few_shots
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")

    def count_tokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        if not string:
            return 0
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))

    @retry_api_call(max_retries=CONFIG["API_CALL_MAX_RETRIES"], initial_delay=CONFIG["API_CALL_INITIAL_DELAY"], max_delay=CONFIG["API_CALL_MAX_DELAY"])
    def generate_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

class TeacherLLM(BaseLLM):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_few_shot: bool = True,
        num_if_few_shots: int = 5,
        recommended_question_token_limit: int = 150,
        recommended_education_theory: Optional[str] = None,
        max_tokens_rerun_threshold_percentage: float = 0.8,
        question_retries: int = 3,
        is_vertex_ai: bool = False,
        project_id: str = None,
        location: str = None,
    ):
        self.is_vertex_ai = is_vertex_ai
        self.project_id = project_id
        self.location = location
        self.token_expiry = 0
        
        if self.is_vertex_ai:
            self.refresh_token()
            base_url = self.client.base_url
            api_key = self.client.api_key
        
        super().__init__(name, model, api_key, base_url, temperature, max_tokens, use_few_shot, num_if_few_shots)
        
        self.recommended_question_token_limit = recommended_question_token_limit
        self.recommended_education_theory = recommended_education_theory
        self.max_tokens_rerun_threshold_percentage = max_tokens_rerun_threshold_percentage
        self.question_retries = question_retries

    def refresh_token(self):
        credentials, _ = google.auth.default()
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        self.client = OpenAI(
            base_url=f'https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi',
            api_key=credentials.token
        )
        self.token_expiry = time.time() + 3540  # Set expiry to 59 minutes from now

    def generate_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.is_vertex_ai and time.time() > self.token_expiry:
            self.refresh_token()
        return super().generate_response(messages)

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            # "api_key": self.api_key,
            # "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_few_shot": self.use_few_shot,
            "num_if_few_shots": self.num_if_few_shots,
            "recommended_question_token_limit": self.recommended_question_token_limit,
            "recommended_education_theory": self.recommended_education_theory,
            "max_tokens_rerun_threshold_percentage": self.max_tokens_rerun_threshold_percentage,
            "question_retries": self.question_retries,
            "is_vertex_ai": self.is_vertex_ai,
            "project_id": self.project_id,
            "location": self.location,
        }

    def generate_question(
        self,
        category: str,
        pre_test_results: List[Dict[str, Any]],
        interaction_history: List[Dict[str, str]],
        current_round: int,
        total_rounds: int,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        system_message = f"""You are an expert teacher in {category} {"using the " + self.recommended_education_theory + " approach " if self.recommended_education_theory else ""}dedicated to enhancing the student's understanding after analyzing the student's response to a pre-test. 
        Your task is to ask {total_rounds} rounds of relevant, thought-provoking questions to the student. 
        You should ask one new question per round (and if needed, provide necessary corrections or feedback for the student's previous round's answers), 
        each under {self.recommended_question_token_limit} tokens, without revealing the correct answers or specific details of the pre-test questions. 
        Your goal is to prepare the student for the post-test by fostering a deeper and more comprehensive understanding of the subject matter.\n\n"""

        few_shot_examples_message = ""
        if self.use_few_shot and few_shot_examples:
            few_shot_examples_message = "\n\nHere are some example questions and reasoning processes:\n"
            for example in few_shot_examples[:self.num_if_few_shots]:
                few_shot_examples_message += f"Question: {example['question']}\nReasoning: {example['cot_content']}\n\n"

        pre_test_info = "\n\nHere are the pre-test results of the student:\n"
        for r in pre_test_results:
            pre_test_info += f"""
                Question ID: {r['question_id']}
                Question: {r['question']}
                Student's Reasoning: {r['model_response']}
                Student's Answer: {r['model_prediction']}
                Student's Answer is Correct or Not: {"Correct." if r['correct_answer'] == r['model_prediction'] else "Incorrect."}

                """

        messages = [{"role": "system", "content": system_message + "\n" + few_shot_examples_message + "\n" + pre_test_info}]

        for interaction in interaction_history:
            messages.append(
                {"role": "assistant", "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', interaction['question'])}"}
            )
            messages.append(
                {"role": "user", "content": f"Student: {re.sub(r'^(Student:( )*)+', '', interaction['answer'])}"}
            )

        messages.append(
            {"role": "user", "content": f"Generate the round {current_round} question ({self.recommended_question_token_limit} tokens or less) to promote better understanding:",}
        )

        # Initialize retry counter, when the output exceeds 80% of max_tokens and is greater than recommended_question_token_limit, a rerun of generation is required.

        retry_count = 0

        while retry_count < self.question_retries:
            response = self.generate_response(messages)
            response_tokens = self.count_tokens(response)
            max_allowed_tokens = max(
                self.max_tokens * self.max_tokens_rerun_threshold_percentage,
                self.recommended_question_token_limit,
            )

            if 0 < response_tokens <= max_allowed_tokens:
                return response
            elif response_tokens == 0:
                logging.warning(f"{self.name}'s {current_round}/{total_rounds} round question for {pre_test_results[0]['question_id']} had 0 tokens (attempt {retry_count + 1}/{self.question_retries}). Retrying...")
            else:
                logging.warning(f"{self.name}'s {current_round}/{total_rounds} round question for {pre_test_results[0]['question_id']} had {response_tokens} tokens and exceeded {max_allowed_tokens} tokens (attempt {retry_count + 1}/{self.question_retries}). Retrying...")
            retry_count += 1

        logging.error(f"{self.name}'s {current_round}/{total_rounds} round question for {pre_test_results[0]['question_id']} exceeded {max_allowed_tokens} tokens after {self.question_retries} attempts. Returning last response.")
        # If after question_retries attempts the response is still too long, return an empty string
        return response

class StudentLLM(BaseLLM):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        answer_max_tokens: int = 1024,
        test_max_tokens: int = 2048,
        use_few_shot: bool = False,
        num_if_few_shots: int = 5,
        include_pretest_info: bool = True,
        recommended_answer_token_limit: int = 150,
        recommended_test_token_limit: int = 1024,
        max_tokens_rerun_threshold_percentage: float = 0.8,
        answer_retries: int = 3,
    ):
        super().__init__(
            name, model, api_key, base_url, temperature, answer_max_tokens, use_few_shot, num_if_few_shots
        )
        self.test_max_tokens = test_max_tokens
        self.include_pretest_info = include_pretest_info
        self.recommended_answer_token_limit = recommended_answer_token_limit
        self.recommended_test_token_limit = recommended_test_token_limit
        self.max_tokens_rerun_threshold_percentage = max_tokens_rerun_threshold_percentage
        self.answer_retries = answer_retries

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            # "api_key": self.api_key,
            # "base_url": self.base_url,
            "temperature": self.temperature,
            "answer_max_tokens": self.max_tokens,
            "test_max_tokens": self.test_max_tokens,
            "use_few_shot": self.use_few_shot,
            "num_if_few_shots": self.num_if_few_shots,
            "include_pretest_info": self.include_pretest_info,
            "recommended_answer_token_limit": self.recommended_answer_token_limit,
            "recommended_test_token_limit": self.recommended_test_token_limit,
            "max_tokens_rerun_threshold_percentage": self.max_tokens_rerun_threshold_percentage,
            "answer_retries": self.answer_retries,
        }
    
    def answer_question(
        self,
        category: str,
        question: str,
        interaction_history: List[Dict[str, str]],
        pre_test_results: Optional[List[Dict[str, Any]]] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        
        system_message = f"You are a student focusing on {category}. Analyze the question carefully, explain your thought process ({self.recommended_answer_token_limit} tokens or less) , and try to apply the concepts you've learned to solve problems. If you're unsure, express your uncertainty and explain your reasoning."

        few_shot_examples_message = ""
        if self.use_few_shot and few_shot_examples:
            few_shot_examples_message = "\n\nHere are some example questions and reasoning processes:\n"
            for example in few_shot_examples[:self.num_if_few_shots]:
                few_shot_examples_message += f"Question: {example['question']}\nReasoning: {example['cot_content']}\n\n"

        
        messages = [{"role": "system", "content": system_message + "\n" + few_shot_examples_message}]

        if self.include_pretest_info and pre_test_results:
            for r in pre_test_results:
                pre_test_question_text = self.format_question(
                    r["question"], r["options"]
                )
                messages.append(
                    {"role": "user", "content": f"Teacher: {pre_test_question_text}"}
                )
                pre_test_model_response = r["model_response"]
                messages.append(
                    {"role": "assistant", "content": f"Student: {pre_test_model_response}"}
                )

        if interaction_history:
            for interaction in interaction_history:
                messages.append(
                    {"role": "user", "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', interaction['question'])}"}
                )
                messages.append(
                    {"role": "assistant", "content": f"Student: {re.sub(r'^(Student:( )*)+', '', interaction['answer'])}"}
                )

        messages.append(
            {"role": "user", "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', question)}\n\nYour thoughtful and detailed answer ({self.recommended_answer_token_limit} tokens or less):"}
        )

        # Initialize retry counter, when the output exceeds 80% of max_tokens and is greater than recommended_question_token_limit, a rerun of generation is required.
        retry_count = 0

        while retry_count < self.answer_retries:
            response = self.generate_response(messages)
            response_tokens = self.count_tokens(response)
            max_allowed_tokens = max(
                self.max_tokens * self.max_tokens_rerun_threshold_percentage,
                self.recommended_answer_token_limit,
            )

            if 0 < response_tokens <= max_allowed_tokens:
                return response
            elif response_tokens == 0:
                logging.warning(f"{self.name}'s {len(interaction_history) + 1} round answer for {pre_test_results[0]['question_id']} had 0 tokens (attempt {retry_count + 1}/{self.answer_retries}). Retrying...")
            else:
                logging.warning(f"{self.name}'s {len(interaction_history) + 1} round answer for {pre_test_results[0]['question_id']} had {response_tokens} tokens and exceeded {max_allowed_tokens} tokens (attempt {retry_count + 1}/{self.answer_retries}). Retrying...")
            retry_count += 1
        logging.error(f"{self.name}'s {len(interaction_history) + 1} round answer for {pre_test_results[0]['question_id']} exceeded {max_allowed_tokens} tokens after {self.answer_retries} attempts. Returning last response.")
        # If after answer_retries attempts the response is still too long, return an empty string
        return response

    def take_test(
        self,
        test_data: List[Dict[str, Any]],
        few_shot_cot_examples: List[Dict[str, Any]] = [],
        interaction_history: Optional[List[Dict[str, str]]] = None,
        pre_test_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for question in test_data:
            category = question["category"]

            system_prompt = (
                "\nThe following are multiple choice questions (with answers) about {}. "
                'Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'.format(
                    category
                )
            )

            if self.use_few_shot:
                system_prompt += "\n\nHere are some example questions and reasoning processes:\n"
                for example in few_shot_cot_examples[:self.num_if_few_shots]:  # Using number of examples from config, adjust as needed
                    system_prompt += self.format_question(
                        example["question"], example["options"], example["cot_content"]
                    )

            messages = [{"role": "system", "content": system_prompt}]

            if self.include_pretest_info and pre_test_results:
                for r in pre_test_results:
                    if r["question_id"] == question["question_id"]:
                        pre_test_question_text = self.format_question(
                            r["question"], r["options"]
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"You previously answered this question: {pre_test_question_text}",
                            }
                        )
                        pre_test_model_response = r["model_response"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"Your previous answer was: {pre_test_model_response}",
                                # "content": f"Student: [MASKED]",
                            }
                        )

            if interaction_history:
                messages.append(
                    {
                        "role": "user",
                        "content": "Teacher: Let's start a conversation about the subject matter.",
                    }
                )
                
                for interaction in interaction_history:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', interaction['question'])}",
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Student: {re.sub(r'^(Student:( )*)+', '', interaction['answer'])}",
                        }
                    )
                
                test_question = f"Based on your current understanding after the conversation with the teacher, rethink step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice:\n"
            else:
                test_question = ""

            test_question += self.format_question(question["question"], question["options"])

            messages.append({"role": "user", "content": test_question})

            try:
                retry_count = 0

                while retry_count < self.answer_retries:
                    start = time.time()
                    response = self.test_call_api(messages)
                    logging.info(f"\n{self.name}'s answer of question {question['question_id']} costs {time.time() - start:.2f} seconds")
                    response_tokens = self.count_tokens(response)
                    max_allowed_tokens = max(
                        self.test_max_tokens * self.max_tokens_rerun_threshold_percentage,
                        self.recommended_test_token_limit,
                    )
                    if 0 < response_tokens <= max_allowed_tokens:
                        break
                    elif response_tokens == 0:
                        logging.warning(f"""{self.name}'s answer of {"posttest" if interaction_history else "pretest"} question {question["question_id"]} had 0 tokens (attempt {retry_count + 1}/{self.answer_retries}). Retrying...""")
                    else:
                        logging.warning(f"""{self.name}'s answer of {"posttest" if interaction_history else "pretest"} question {question["question_id"]} had {response_tokens} tokens and exceeded {max_allowed_tokens} tokens (attempt {retry_count + 1}/{self.answer_retries}). Retrying...""")
                    retry_count += 1
                else:
                    logging.error(f"""{self.name}'s answer of {"posttest" if interaction_history else "pretest"} question {question['question_id']} exceeded {max_allowed_tokens} tokens after {self.answer_retries} attempts. Using last response.""")
                    # return response
            except Exception as e:
                logging.error(f"Error: {e}")
                response = " "
            prediction = self.extract_answer(response)

            results.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "options": question["options"],
                    "correct_answer": question.get("correct_answer", question.get("answer")),
                    "correct_answer_index": question.get("correct_answer_index", question.get("answer_index")),
                    "model_response": response,
                    "model_prediction": prediction,
                    "category": category,
                }
            )
        return results

    @retry_api_call(max_retries=CONFIG["API_CALL_MAX_RETRIES"], initial_delay=CONFIG["API_CALL_INITIAL_DELAY"], max_delay=CONFIG["API_CALL_MAX_DELAY"])
    def test_call_api(self, messages):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=self.test_max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return response.choices[0].message.content

    @staticmethod
    def format_question(
        question: str, options: List[str], cot_content: str = ""
    ) -> str:
        if cot_content == "":
            cot_content = "Let's think step by step."
        if cot_content.startswith("A: "):
            cot_content = cot_content[3:]
        example = f"Question: {question}\nOptions: "
        choice_map = "ABCDEFGHIJ"
        for i, opt in enumerate(options):
            example += f"{choice_map[i]}. {opt}\n"
        example += f"Answer: {cot_content}\n\n"
        return example

    @staticmethod
    def extract_answer(text: str) -> str:
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            match = re.search(r".*[aA]nswer:\s*([A-J])", text)
            if match:
                return match.group(1)
            else:
                pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
                match = re.search(pattern, text, re.DOTALL)
                return match.group(0) if match else "R"

class EvaluatorLLM(BaseLLM):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_few_shot: bool = False,
        num_if_few_shots: int = 5,
    ):
        super().__init__(name, model, api_key, base_url, temperature, max_tokens, use_few_shot, num_if_few_shots)
        self.interaction_analysis_dimensions = [
            "Assessment Effectiveness",
            "Questioning Effectiveness",
            "Feedback Effectiveness",
            "Instructional Adaptation Effectiveness",
            "Learning Objective Achievement Effectiveness"
        ]
        self.teacher_questions_analysis_dimensions = [
            "Question Relevance",
            "Cognitive Level",
            "Knowledge Dimension",
            "Question Diversity",
            "Scaffolding Progression",
            "Metacognitive Promotion"
        ]
        self.student_responses_analysis_dimensions = [
            "Response Relevance",
            "Cognitive Level Demonstration",
            "Knowledge Dimension Integration",
            "Response Diversity",
            "Elaboration Progression",
            "Metacognitive Reflection"
        ]
        self.interaction_analysis_schema = self.create_dimension_schema(self.interaction_analysis_dimensions)
        self.teacher_questions_analysis_schema = self.create_dimension_schema(self.teacher_questions_analysis_dimensions)
        self.student_responses_analysis_schema = self.create_dimension_schema(self.student_responses_analysis_dimensions)
    
    def create_dimension_schema(self, dimensions):
        return {
            "type": "object",
            "properties": {
                "teacher_a": {
                    "type": "object",
                    "properties": {
                        dimension: {
                            "type": "object",
                            "properties": {
                                "analysis": {"type": "string"},
                                "score": {"type": "number"}
                            },
                            "required": ["analysis", "score"],
                            "additionalProperties": False
                        } for dimension in dimensions
                    },
                    "required": dimensions,
                    "additionalProperties": False
                },
                "teacher_b": {
                    "type": "object",
                    "properties": {
                        dimension: {
                            "type": "object",
                            "properties": {
                                "analysis": {"type": "string"},
                                "score": {"type": "number"}
                            },
                            "required": ["analysis", "score"],
                            "additionalProperties": False
                        } for dimension in dimensions
                    },
                    "required": dimensions,
                    "additionalProperties": False
                },
                "verdict": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string"},
                        "choice": {
                            "type": "string",
                            "enum": ["A", "B", "C"]
                        }
                    },
                    "required": ["analysis", "choice"],
                    "additionalProperties": False
                }
            },
            "required": ["teacher_a", "teacher_b", "verdict"],
            "additionalProperties": False
        }
    
    def calculate_accuracy(self, test_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        category_scores = defaultdict(lambda: {"correct": 0, "total": 0})
        total_correct = 0
        total_questions = len(test_responses)

        for response in test_responses:
            category = response["category"]
            category_scores[category]["total"] += 1
            if response["correct_answer"] == response["model_prediction"]:
                category_scores[category]["correct"] += 1
                total_correct += 1

        category_accuracy = {
            category: scores["correct"] / scores["total"]
            for category, scores in category_scores.items()
        }
        overall_accuracy = total_correct / total_questions

        return {
            "category_accuracy": category_accuracy,
            "overall_accuracy": overall_accuracy,
        }
    
    def teacher_questions_analysis(
        self,
        question_id: str,
        category: str,
        teacher_1_name: str,
        teacher_1_interaction: List[Dict[str, str]],
        teacher_2_name: str,
        teacher_2_interaction: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        # Randomly decide which teacher will be 'teacher_a' in the prompt
        if random.choice([True, False]):
            prompt_teacher_a = (teacher_1_name, teacher_1_interaction)
            prompt_teacher_b = (teacher_2_name, teacher_2_interaction)
        else:
            prompt_teacher_a = (teacher_2_name, teacher_2_interaction)
            prompt_teacher_b = (teacher_1_name, teacher_1_interaction)

        # Create a mapping for later de-anonymization
        teacher_map = {
            "teacher_a": prompt_teacher_a[0],
            "teacher_b": prompt_teacher_b[0]
        }

        instruction = f"""
You are an expert in educational assessment with a deep understanding of learning theories and pedagogical practices. Your task is to evaluate the teaching effectiveness of two teachers based on their questions of interactions with a student. Please consider the following six dimensions in your evaluation:

1. Question Relevance: 
Assess how well the questions target key learning goals and address critical areas of student understanding or misunderstanding.

2. Cognitive Level: 
Evaluate the distribution and advancement of questions across different levels of cognitive complexity (remembering, understanding, applying, analyzing, evaluating, creating).

3. Knowledge Dimension: 
Assess how well the questions cover and integrate different dimensions of knowledge (factual, conceptual, procedural, metacognitive), promoting comprehensive understanding.

4. Question Diversity: 
Evaluate the teacher's use of various question types (e.g., Playground, Brainstorm, Focal, General Invitation, Lower-level Divergent, Analytic Convergent, Shotgun/Funnel) to stimulate diverse cognitive processes.

5. Scaffolding Progression: 
Assess how well the sequence of questions builds upon previous responses, incrementally increasing in complexity while providing necessary support.

6. Metacognitive Promotion: 
Evaluate how effectively questions prompt students to reflect on their own thinking processes, learning strategies, and self-regulation.


**Instructions:**

1. Evaluate each teacher across the six dimensions listed in the schema.
2. For each dimension, provide:
   - An `analysis` string that explains your step by step evaluation.
   - A `score` from 1 to 10 (1 being the lowest, 10 being the highest).
3. Provide an overall `verdict`:
   - An `analysis` string that explains your step by step final judgment.
   - A `choice` that is `"A"` if teacher_a is better overall, `"B"` if teacher_b is better overall, and `"C"` if their performance is equally effective.
"""

        inputs = f"""Question ID: {question_id}
Category: {category}

<|The Start of teacher_a's Questions of Interaction with Student|>
{self.format_teacher_questions(prompt_teacher_a[1])}
<|The End of teacher_a's Questions of Interaction with Student|>

<|The Start of teacher_b's Questions of Interaction with Student|>
{self.format_teacher_questions(prompt_teacher_b[1])}
<|The End of teacher_b's Questions of Interaction with Student|>

Please provide your evaluation of both teachers:
"""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inputs},
        ]

        evaluation = self.generate_response(messages, schema=self.teacher_questions_analysis_schema)
        if evaluation is None:
            logging.error(f"Failed to generate evaluation for question {question_id}. Returning empty evaluation.")
            return {}

        # De-anonymize the response
        evaluation = self.deanonymize_evaluation(evaluation, teacher_map)
        
        parsed_evaluation = self.parse_evaluation(evaluation)
        
        return parsed_evaluation

    def student_responses_analysis(
        self,
        question_id: str,
        category: str,
        teacher_1_name: str,
        teacher_1_interaction: List[Dict[str, str]],
        teacher_2_name: str,
        teacher_2_interaction: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        # Randomly decide which teacher will be 'teacher_a' in the prompt
        if random.choice([True, False]):
            prompt_teacher_a = (teacher_1_name, teacher_1_interaction)
            prompt_teacher_b = (teacher_2_name, teacher_2_interaction)
        else:
            prompt_teacher_a = (teacher_2_name, teacher_2_interaction)
            prompt_teacher_b = (teacher_1_name, teacher_1_interaction)

        # Create a mapping for later de-anonymization
        teacher_map = {
            "teacher_a": prompt_teacher_a[0],
            "teacher_b": prompt_teacher_b[0]
        }

        instruction = f"""
You are an expert in educational assessment with a deep understanding of learning theories and pedagogical practices. Your task is to evaluate the teaching effectiveness of two teachers based on the performance of the student with the interactions of each teacher. Please consider the following six dimensions in your evaluation:

1. Response Relevance: 
Assess how well student responses address the key concepts and learning goals targeted by the teacher's questions.

2. Cognitive Level Demonstration: 
Evaluate the cognitive complexity of student answers (remembering, understanding, applying, analyzing, evaluating, creating) and how this complexity evolves over the course of the interaction.

3. Knowledge Dimension Integration: 
Assess how students incorporate and connect different forms of knowledge (factual, conceptual, procedural, metacognitive) in their answers, demonstrating comprehensive understanding.

4. Response Diversity: 
Evaluate students' ability to approach questions from multiple angles and provide diverse explanations or problem-solving approaches.

5. Elaborating Progression: 
Assess how student answers evolve in terms of depth, complexity, and sophistication throughout the questioning sequence.

6. Metacognitive Reflection: 
Evaluate how students reflect on their own thinking processes, learning strategies, and self-assessment in their answers.


**Instructions:**

1. Evaluate each teacher across the six dimensions listed in the schema.
2. For each dimension, provide:
   - An `analysis` string that explains your step by step evaluation.
   - A `score` from 1 to 10 (1 being the lowest, 10 being the highest).
3. Provide an overall `verdict`:
   - An `analysis` string that explains your step by step final judgment.
   - A `choice` that is `"A"` if student's performance under teacher_a is better overall, `"B"` if under teacher_b is better overall, and `"C"` if their performance is equally effective.
"""

        inputs = f"""Question ID: {question_id}
Category: {category}

<|The Start of student's Answers under teacher_a's Questions|>
{self.format_student_responses(prompt_teacher_a[1])}
<|The End of student's Answers under teacher_a's Questions|>

<|The Start of student's Answers under teacher_b's Questions|>
{self.format_student_responses(prompt_teacher_b[1])}
<|The End of student's Answers under teacher_b's Questions|>

Please provide your evaluation of both teachers:
"""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inputs},
        ]

        evaluation = self.generate_response(messages, schema=self.student_responses_analysis_schema)
        if evaluation is None:
            logging.error(f"Failed to generate evaluation for question {question_id}. Returning empty evaluation.")
            return {}

        # De-anonymize the response
        evaluation = self.deanonymize_evaluation(evaluation, teacher_map)
        
        parsed_evaluation = self.parse_evaluation(evaluation)
        
        return parsed_evaluation

    def over_interaction_analysis(
        self,
        question_id: str,
        category: str,
        pre_test_result: Dict[str, Any],
        teacher_1_name: str,
        teacher_1_interaction: List[Dict[str, str]],
        teacher_2_name: str,
        teacher_2_interaction: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        # Randomly decide which teacher will be 'teacher_a' in the prompt
        if random.choice([True, False]):
            prompt_teacher_a = (teacher_1_name, teacher_1_interaction)
            prompt_teacher_b = (teacher_2_name, teacher_2_interaction)
        else:
            prompt_teacher_a = (teacher_2_name, teacher_2_interaction)
            prompt_teacher_b = (teacher_1_name, teacher_1_interaction)

        # Create a mapping for later de-anonymization
        teacher_map = {
            "teacher_a": prompt_teacher_a[0],
            "teacher_b": prompt_teacher_b[0]
        }

        instruction = f"""
You are an expert in educational assessment with a deep understanding of learning theories and pedagogical practices. Your task is to evaluate the teaching effectiveness of two teachers based on their interactions with a student. Please consider the following five dimensions in your evaluation:

1. Assessment Effectiveness: 
Evaluates how well the teacher identifies students' current understanding and learning gaps, and how effectively they use this information to inform their teaching approach.

2. Questioning Effectiveness: 
Assesses how well the teacher's questions stimulate reflection, clarify concepts, challenge misconceptions, and deepen understanding.

3. Feedback Effectiveness: 
Evaluates the quality and impact of teacher feedback on student understanding, motivation, and self-regulated learning.

4. Instructional Adaptation Effectiveness: 
Assesses how well the teacher modifies strategies, content, pace, and difficulty in real-time to meet student needs and enhance learning.

5. Learning Objective Achievement Effectiveness: 
Evaluates how well the entire teaching interaction promotes conceptual understanding, skill development, and metacognitive ability enhancement.


**Instructions:**

1. Evaluate each teacher across the five dimensions listed in the schema.
2. For each dimension, provide:
   - An `analysis` string that explains your step by step evaluation.
   - A `score` from 1 to 10 (1 being the lowest, 10 being the highest).
3. Provide an overall `verdict`:
   - An `analysis` string that explains your step by step final judgment.
   - A `choice` that is `"A"` if teacher_a is better overall, `"B"` if teacher_b is better overall, and `"C"` if their performance is equally effective.
"""

        inputs = f"""Question ID: {question_id}
Category: {category}

Pre-test Result:
{json.dumps(pre_test_result, indent=2)}

<|The Start of teacher_a's Interaction with Student|>
{self.format_interaction(prompt_teacher_a[1])}
<|The End of teacher_a's Interaction with Student|>

<|The Start of teacher_b's Interaction with Student|>
{self.format_interaction(prompt_teacher_b[1])}
<|The End of teacher_b's Interaction with Student|>

Please provide your evaluation of both teachers:
"""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inputs},
        ]

        evaluation = self.generate_response(messages, schema=self.interaction_analysis_schema)
        if evaluation is None:
            logging.error(f"Failed to generate evaluation for question {question_id}. Returning empty evaluation.")
            return {}

        # De-anonymize the response
        evaluation = self.deanonymize_evaluation(evaluation, teacher_map)
        
        parsed_evaluation = self.parse_evaluation(evaluation)
        
        return parsed_evaluation

    def format_interaction(self, interaction: List[Dict[str, str]]) -> str:
        formatted = ""
        for turn in interaction:
            formatted += f"### Teacher:\n{re.sub(r'^(Teacher:( )*)+', '', turn['question'])}\n### Student:\n{re.sub(r'^(Student:( )*)+', '', turn['answer'])}\n"
            # f"Teacher: {re.sub(r'^(Teacher:( )*)+', '', interaction['question'])}"
        return formatted
    
    def format_teacher_questions(self, interaction: List[Dict[str, str]]) -> str:
        formatted = ""
        for turn in interaction:
            formatted += f"### Teacher:\n{re.sub(r'^(Teacher:( )*)+', '', turn['question'])}\n"
        return formatted
    
    def format_student_responses(self, interaction: List[Dict[str, str]]) -> str:
        formatted = ""
        for turn in interaction:
            formatted += f"### Student:\n{re.sub(r'^(Student:( )*)+', '', turn['answer'])}\n"
        return formatted

    def parse_evaluation(self, evaluation: str) -> Dict[str, Any]:
        try:
            evaluation_json = json.loads(evaluation)
            return evaluation_json
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
            return {}

    def deanonymize_evaluation(self, evaluation: str, teacher_map: Dict[str, str]) -> str:
        for anonymous_name, actual_name in teacher_map.items():
            # Replace the teacher keys
            evaluation = evaluation.replace(f'"{anonymous_name}":', f'"{actual_name}":')
            
            # Replace teacher references in the analysis texts
            evaluation = evaluation.replace(f'Teacher {anonymous_name[-1]}', actual_name)
        
        # Replace the choice
        choice_map = {
            '"choice": "A"': f'"choice": "{teacher_map["teacher_a"]}"',
            '"choice":"A"': f'"choice":"{teacher_map["teacher_a"]}"',
            '"choice": "B"': f'"choice": "{teacher_map["teacher_b"]}"',
            '"choice":"B"': f'"choice":"{teacher_map["teacher_b"]}"',
            '"choice": "C"': '"choice": "Tie"',
            '"choice":"C"': '"choice":"Tie"'
        }
        for anonymous_choice, actual_choice in choice_map.items():
            evaluation = evaluation.replace(anonymous_choice, actual_choice)
        
        return evaluation
    
    def generate_response(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation_response",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        return response.choices[0].message.content


# BaseDataset class
class BaseDataset:
    @classmethod
    def load_data(cls, dataset_name: str):
        raise NotImplementedError

    @classmethod
    def preprocess_data(cls, data):
        raise NotImplementedError

    @classmethod
    def filter_data(
        cls,
        data: Dict[str, List[Dict[str, Any]]],
        selected_question_ids: Optional[List[str]],
        selected_categories: Optional[List[str]],
        first_questions_size: Optional[int],
        questions_sample_size: Optional[int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        filtered_data = {}

        # Convert selected_question_ids to strings
        if selected_question_ids:
            selected_question_ids = [str(qid) for qid in selected_question_ids]

        # If both selected_question_ids and selected_categories are empty, use all categories
        categories_to_use = (
            selected_categories if selected_categories else list(data.keys())
        )

        if selected_question_ids:
            for category, questions in data.items():
                filtered_questions = [
                    q for q in questions if str(q["question_id"]) in selected_question_ids
                ]
                if filtered_questions:
                    filtered_data[category] = filtered_questions
        else:
            for category in categories_to_use:
                if category in data:
                    questions = data[category]
                    if first_questions_size is not None:
                        filtered_data[category] = questions[:first_questions_size]
                    elif questions_sample_size is not None:
                        filtered_data[category] = random.sample(
                            questions, min(questions_sample_size, len(questions))
                        )
                    else:
                        filtered_data[category] = questions

        # If all filtering parameters are empty or None, return the original test_data
        if (
            not selected_question_ids
            and not selected_categories
            and first_questions_size is None
            and questions_sample_size is None
        ):
            return data

        logging.info(
            f"Filtered data: {len(filtered_data)} categories, {sum(len(q) for q in filtered_data.values())} questions"
        )
        return filtered_data

class MMLU_PRO(BaseDataset):
    @classmethod
    def load_data(cls, dataset_name: str = "TIGER-Lab/MMLU-Pro"):
        dataset = load_dataset(dataset_name)
        test_df, val_df = dataset["test"], dataset["validation"]
        test_df = cls.preprocess_data(test_df)
        val_df = cls.preprocess_data(val_df)
        return test_df, val_df

    @classmethod
    def preprocess_data(cls, data):
        categorized_data = defaultdict(list)
        for entry in data:
            categorized_data[entry["category"]].append(
                {
                    "question_id": str(entry["question_id"]),
                    "question": entry["question"],
                    "options": [opt for opt in entry["options"] if opt != "N/A"],
                    "answer": entry["answer"],
                    "answer_index": entry["answer_index"],
                    "cot_content": entry["cot_content"],
                    "category": entry["category"],
                }
            )
        return dict(categorized_data)

class GPQA(BaseDataset):
    SEED = 42
    @classmethod
    def load_data(cls, test_data_folder_path: str, val_data_filepath: str, dataset_name: str = "gpqa_diamond.csv"):
        random.seed(cls.SEED)

        test_df = pd.read_csv(os.path.join(test_data_folder_path, dataset_name))
        test_df = cls.preprocess_data(test_df)

        with open(val_data_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        val_df = defaultdict(list)

        for question in data["questions"]:
            options = list(question["choices"].values())
            correct_answer = question["choices"][question["correct_answer"]]
            val_df["general"].append(
                {
                    "question_id": str(hash(question["question"])),
                    "question": question["question"],
                    "options": options,
                    "answer": correct_answer,
                    "answer_index": options.index(correct_answer),
                    "cot_content": question["explanation"],
                    "category": "general",
                }
            )

        val_df = dict(val_df)

        return test_df, val_df

    @classmethod
    def preprocess_data(cls, data: pd.DataFrame):
        INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
        categorized_data = defaultdict(list)
        for _, row in data.iterrows():
            options = [
                row["Correct Answer"], 
                row["Incorrect Answer 1"], 
                row["Incorrect Answer 2"], 
                row["Incorrect Answer 3"]
            ]
            random.shuffle(options)
            correct_index = options.index(row["Correct Answer"])

            categorized_data[row["High-level domain"]].append(
                {
                    "question_id": str(row["Record ID"]),
                    "question": row["Question"],
                    "options": options,
                    "answer": INDEX_TO_LETTER[correct_index],
                    "answer_index": correct_index,
                    "cot_content": row["Explanation"],
                    "category": row["High-level domain"],
                }
            )
        return dict(categorized_data)
    
class AGIEVAL(BaseDataset):
    @classmethod
    def load_data(cls, test_data_floder_path: str, val_data_filepath: str, dataset_names: List[str] = ["sat-math", "aqua-rat"]):
        """
        Load AGIEval datasets from JSONL files and preprocess them.

        Args:
            dataset_folder_path (str): Path to the folder containing AGIEval JSONL files.
            val_data_filepath (str): Path to the few_shot_prompts.csv file for validation data.
            dataset_names (List[str], optional): List of dataset names to load. Defaults to ["sat-math", "aqua-rat"].

        Returns:
            Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]: A tuple containing test and validation data.
        """
        test_df = {}
        for dataset_name in dataset_names:
            file_path = os.path.join(test_data_floder_path, f"{dataset_name}.jsonl")
            if not os.path.exists(file_path):
                logging.error(f"File {file_path} does not exist.")
                continue
            raw_data = cls.read_jsonl(file_path)
            preprocessed_data = cls.preprocess_data(raw_data, dataset_name)
            # test_df[dataset_name] = preprocessed_data
            test_df.update(preprocessed_data)

        val_df = cls.load_val_data(val_data_filepath, dataset_names)
        return test_df, val_df

    @classmethod
    def read_jsonl(cls, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f] 
    
    @classmethod
    def preprocess_data(cls, data: List[Dict[str, Any]], dataset_name: str):
        """
        Preprocess AGIEval dataset by formatting questions and stripping option prefixes.

        Args:
            data (List[Dict[str, Any]]): Raw data loaded from JSONL.
            dataset_name (str): Name of the dataset being processed.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Preprocessed data categorized by dataset name.
        """
        categorized_data = defaultdict(list)
        for index, entry in enumerate(data):
            question_id = f"{dataset_name}-{index}"
            question = cls.strip_option_prefix(entry["question"])
            options = [cls.strip_option_prefix(opt) for opt in entry.get("options", [])]
            label = entry.get("label", entry.get("answer", ""))
            answer_index = cls.get_answer_index(label)
            passage = entry.get("passage", "")
            cot_content = entry.get("explanation", "")

            categorized_data[dataset_name].append({
                "question_id": question_id,
                "question": question,
                "options": options,
                "answer": label,
                "answer_index": answer_index,
                "cot_content": cot_content,
                "category": dataset_name,
                "passage": passage,
            })
        return dict(categorized_data)
    
    @staticmethod
    def strip_option_prefix(option: str) -> str:
        """
        Remove option prefixes like (A), (B), etc., from the options.

        Args:
            option (str): The option string.

        Returns:
            str: The option string without the prefix.
        """
        return re.sub(r'^\([A-Z]\)\s*', '', option).strip()

    @staticmethod
    def get_answer_index(label: str) -> int:
        """
        Get the index of the correct answer based on the label.

        Args:
            label (str): The label of the correct answer (e.g., 'A', 'B', etc.).

        Returns:
            int: The index of the correct answer.
        """
        label = label.upper()
        try:
            return ord(label) - ord('A')
        except:
            return -1
        
    @classmethod
    def load_val_data(cls, val_data_filepath: str, dataset_names: List[str]):
        """
        Load validation (few-shot) data from few_shot_prompts.csv.

        Args:
            val_data_filepath (str): Path to the few_shot_prompts.csv file.
            dataset_names (List[str]): List of dataset names to include.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Validation data categorized by dataset name.
        """
        val_df = defaultdict(list)
        raw_prompts = pd.read_csv(val_data_filepath, index_col=0)

        for dataset_name in dataset_names:
            if dataset_name not in raw_prompts.columns:
                logging.warning(f"Dataset {dataset_name} not found in few_shot_prompts.csv")
                continue
            
            samples = raw_prompts[raw_prompts.index.str.contains('sample', case=False)][dataset_name]
            explanations = raw_prompts[raw_prompts.index.str.contains('explanation', case=False)][dataset_name]

            for idx, (sample, explanation) in enumerate(zip(samples, explanations), start=1):

                if pd.isna(sample) or pd.isna(explanation):
                    continue

                try:
                    sample_data = ast.literal_eval(sample)
                    options = [cls.strip_option_prefix(opt) for opt in sample_data.get("options", [])]
                    label = sample_data.get("label", "")
                    val_df[dataset_name].append({
                        "question_id": f"{dataset_name}_fewshot_{idx}",
                        "passage": sample_data.get("passage", ""),
                        "question": sample_data.get("question", ""),
                        "options": options,
                        "answer": label,
                        "answer_index": cls.get_answer_index(label),
                        "cot_content": explanation,
                        "category": dataset_name,
                    })
                except (ValueError, SyntaxError) as e:
                    logging.warning(f"Error parsing data for {dataset_name}, sample {idx}: {str(e)}")
                except Exception as e:
                    logging.warning(f"Unexpected error processing sample for {dataset_name}, sample {idx}: {str(e)}")
        return dict(val_df)
      

class EvalManager:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.teachers = self._create_teachers()
        self.students = self._create_students()
        self.evaluator = self._create_evaluator()
        #self.datasets = self._load_datasets()
        self.test_data, self.val_data = self._load_datasets()

    def _create_teachers(self):
        return [TeacherLLM(**config, num_if_few_shots=self.config.num_if_few_shots) for config in self.config.teacher_configs]

    def _create_students(self):
        return [StudentLLM(**config, num_if_few_shots=self.config.num_if_few_shots) for config in self.config.student_configs]

    def _create_evaluator(self):
        return EvaluatorLLM(**self.config.evaluator_config)

    def _load_datasets(self):
        if self.config.dataset_type == "mmlu-pro":
            try:
                test_data, val_data = MMLU_PRO.load_data(self.config.dataset_name)
                filtered_test_data = MMLU_PRO.filter_data(
                test_data,
                self.config.selected_question_ids,
                self.config.selected_categories,
                self.config.first_questions_size,
                self.config.questions_sample_size,
            )
                return filtered_test_data, val_data
            except Exception as e:
                logging.error(f"Error loading MMLU-PRO data: {str(e)}")
                raise
        elif self.config.dataset_type == "gpqa":
            if not self.config.gpqa_test_data_folder_path or not self.config.gpqa_val_data_filepath:
                raise ValueError("GPQA test data folder path and validation data filepath must be provided.")
            try:
                test_data, val_data = GPQA.load_data(self.config.gpqa_test_data_folder_path, self.config.gpqa_val_data_filepath, self.config.dataset_name)

                filtered_test_data = GPQA.filter_data(
                    test_data,
                    self.config.selected_question_ids,
                    self.config.selected_categories,
                    self.config.first_questions_size,
                    self.config.questions_sample_size,
                )
                return filtered_test_data, val_data
            except Exception as e:
                logging.error(f"Error loading GPQA data: {str(e)}")
                raise
        elif self.config.dataset_type == "agieval":
            if not self.config.agieval_test_data_folder_path or not self.config.agieval_val_data_filepath:
                raise ValueError("AGIEval test data folder path and validation data filepath must be provided.")
            try:
                test_data, val_data = AGIEVAL.load_data(
                    self.config.agieval_test_data_folder_path,
                    self.config.agieval_val_data_filepath,
                    self.config.agieval_dataset_names
                )
                filtered_test_data = AGIEVAL.filter_data(
                    test_data,
                    self.config.selected_question_ids,
                    self.config.selected_categories,
                    self.config.first_questions_size,
                    self.config.questions_sample_size,
                )
                return filtered_test_data, val_data
            except Exception as e:
                logging.error(f"Error loading AGIEval data: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")

    def _get_teacher_api_info(self, teacher_name, model_name):
        for teacher_config in self.config.teacher_configs:
            if teacher_config['name'] == teacher_name:
                return teacher_config['api_key'], teacher_config['base_url']
        raise ValueError(f"Teacher {teacher_name} not found in configuration")

    def _get_student_api_info(self, student_name, student_model):
        for student_config in self.config.student_configs:
            if student_config['name'] == student_name and student_config['model'] == student_model:
                return student_config['api_key'], student_config['base_url']
        raise ValueError(f"No matching API info found for student {student_name} with model {student_model}") 

    def run_complete_eval(self):
        pretest_results = self._run_pretest()
        interaction_results = self._run_interactions(pretest_results)
        posttest_results = self._run_posttest(interaction_results)
        evaluation_results = self._perform_evaluation(posttest_results)
        return evaluation_results

    def _run_pretest(self):
        pretest_results = {}
        for student in self.students:
            student_name = student.name
            student_results = {}
            with ThreadPoolExecutor(max_workers=CONFIG["PARALLEL_TASKS"]) as executor:
                future_to_question = {}
                for category, questions in self.test_data.items():
                    few_shot_cot_examples = self.val_data.get(category, self.val_data.get("general", []))
                    for question in questions:
                        question_id = str(question["question_id"])
                        future = executor.submit(
                            student.take_test, [question], few_shot_cot_examples
                        )
                        future_to_question[future] = (category, question_id, student_name)

                for future in tqdm(
                    as_completed(future_to_question),
                    total=len(future_to_question),
                    desc=f"Pretest for {student_name}",
                ):
                    category, question_id, student_name = future_to_question[future]
                    try:
                        result = future.result()
                        scores = self.evaluator.calculate_accuracy(result)
                        student_results[str(question_id)] = {
                            "category": category,
                            "responses": result,
                            "scores": scores,
                        }
                    except Exception as e:
                        logging.error(f"Error in pretest for student {student_name}, question {question_id}: {str(e)}")
                        student_results[str(question_id)] = {
                            "category": category,
                            "responses": [],
                            "scores": {"category_accuracy": {}, "overall_accuracy": 0},
                            "error": str(e)
                        }

            # Check for missing questions and questions with empty responses
            all_question_ids = set(str(q["question_id"]) for questions in self.test_data.values() for q in questions)
            missing_question_ids = all_question_ids - set(student_results.keys())
            empty_response_question_ids = set(qid for qid, data in student_results.items() if not data["responses"])
            question_ids_to_rerun = missing_question_ids.union(empty_response_question_ids)

            #Rerun the questions that are missing or have empty responses
            for attempt in range(student.answer_retries):
                if not question_ids_to_rerun:
                    break
                logging.warning(f"Rerun attempt {attempt + 1} for {len(question_ids_to_rerun)} questions.")

                for question_id in list(question_ids_to_rerun):
                    question = next((q for cat in self.test_data.values() for q in cat if str(q["question_id"]) == question_id), None)
                    if question is None:
                        logging.error(f"Question with ID {question_id} not found in test_data")
                        question_ids_to_rerun.remove(question_id)
                        continue
                    try:
                        few_shot_cot_examples = self.val_data.get(question["category"], self.val_data.get("general", []))
                        result = student.take_test([question], few_shot_cot_examples)
                        if result:
                            scores = self.evaluator.calculate_accuracy(result)
                            student_results[str(question_id)] = {
                            "category": question["category"],
                                "responses": result,
                                "scores": scores,
                            }
                            question_ids_to_rerun.remove(question_id)
                    except Exception as e:
                        logging.error(f"Error in rerun attempt {attempt + 1} for question {question_id}: {str(e)}")
                logging.error(f"Rerun attempt {attempt + 1} completed. {len(question_ids_to_rerun)} questions still need processing.")

        # Add empty results for any remaining questions
            for question_id in question_ids_to_rerun:
                question = next((q for cat in self.test_data.values() for q in cat if str(q["question_id"]) == question_id), None)
                if question:
                    student_results[str(question_id)] = {
                        "category": question["category"],
                        "responses": [],
                        "scores": {"category_accuracy": {}, "overall_accuracy": 0},
                        "error": "Failed to process after multiple attempts"
                    }
                else:
                    logging.error(f"Question with ID {question_id} not found in test_data when adding empty results")


            pretest_results[student_name] = {
                "config": student.get_config_dict(),
                "results": student_results,  
            }

        self._save_results(pretest_results, "pretest_results")
        return pretest_results

    def _run_interactions(self, pretest_results):
        interaction_results = defaultdict(lambda: defaultdict(dict))
        progress_file = os.path.join(self.config.output_path, f"progress_{self.config.experiment_version}.json")
        progress_results_file = os.path.join(self.config.output_path, f"progress_results_{self.config.experiment_version}.json")
        progress = load_progress(progress_file)

        # Load existing results if any
        if os.path.exists(progress_results_file):
            with open(progress_results_file, 'r') as f:
                interaction_results = json.load(f)

        for teacher in self.teachers:
            teacher_name = teacher.name
            teacher_config = teacher.get_config_dict()
            interaction_results[teacher.name]['config'] = teacher_config
            
            for student_name, student_data in pretest_results.items():
                # 检查这一组师生是否已经完成
                if f"{teacher_name}_{student_name}" in progress:
                    logging.info(f"Skipping completed teacher-student pair: {teacher_name}_{student_name}")
                    continue
                
                student_config = student_data['config']
                interaction_results[teacher.name][student_name]['config'] = student_config
                api_key, base_url = self._get_student_api_info(student_name, student_config['model'])
                student = StudentLLM(**student_config, api_key=api_key, base_url=base_url)

                student_results = {}
                
                with ThreadPoolExecutor(max_workers=CONFIG["PARALLEL_TASKS"]) as executor:
                    future_to_question = {}
                    
                    for category, questions in self.test_data.items():
                        few_shot_cot_examples = self.val_data.get(category, self.val_data.get("general", []))
                        for question in questions:
                            question_id = str(question['question_id'])
                            if question_id not in student_data['results']:
                                logging.error(f"Question {question_id} not found in pretest results for student {student_name}. Skipping.")
                                continue

                            pretest_result = student_data['results'][question_id]
                            future = executor.submit(
                                self._process_question,
                                teacher,
                                student,
                                question,
                                few_shot_cot_examples,
                                self.config.num_interactions,
                                pretest_result['responses'],
                                pretest_result['scores']
                            )
                            future_to_question[future] = (teacher_name, student_name, question_id)

                    for future in tqdm(
                        as_completed(future_to_question),
                        total=len(future_to_question),
                        desc=f"Interactions for {student_name} with {teacher.name}"
                    ):
                        teacher_name, student_name, question_id = future_to_question[future]
                        try:
                            question_results = future.result()
                            student_results[question_id] = question_results
                        except Exception as e:
                            logging.error(f"Error processing question {question_id} with {teacher.name} and {student_name}: {str(e)}")
                
                # 将这一组师生的结果添加到总结果中
                interaction_results[teacher_name][student_name].update(student_results)
                
                # 更新进度并保存结果
                progress[f"{teacher_name}_{student_name}"] = True
                
                # 一次性保存整个师生组的结果和进度
                with open(progress_results_file, 'w') as f:
                    json.dump(interaction_results, f, indent=2)
                save_progress(progress, progress_file)

                logging.info(f"Completed and saved results for teacher {teacher_name} and student {student_name}")

        self._save_results(dict(interaction_results), "pretest_interaction_results")
        return dict(interaction_results)

    def _process_question(self, teacher, student, question, few_shot_cot_examples, num_interactions, pre_test_result, pre_test_score):
        category = question["category"]
        question_id = str(question["question_id"])

        interaction_history = []
        for i in range(num_interactions):
            teacher_question = teacher.generate_question(
                category,
                pre_test_result,
                interaction_history,
                i + 1,
                num_interactions,
                few_shot_cot_examples if teacher.use_few_shot else None,
            )
            student_answer = student.answer_question(
                category, 
                teacher_question, 
                interaction_history, 
                pre_test_result, 
                few_shot_cot_examples if student.use_few_shot else None,
            )
            interaction_history.append(
                {"question": re.sub(r'^(Teacher:( )*)+', '', teacher_question), "answer": re.sub(r'^(Student:( )*)+', '', student_answer)}
            )

        return {
            "category": category,
            "question_id": question_id,
            "pre_test": {"responses": pre_test_result, "scores": pre_test_score},
            "interaction": interaction_history,
        }
    
    def _run_interaction_from_json(self, pretest_results_path: str):
        with open(pretest_results_path, "r") as f:
            pretest_results = json.load(f)
        return self._run_interactions(pretest_results)

    def _run_posttest(self, interaction_results):
        posttest_results = defaultdict(lambda: defaultdict(dict))

        for teacher_name, teacher_data in interaction_results.items():
            posttest_results[teacher_name]['config'] = teacher_data['config']
            
            for student_name, student_data in teacher_data.items():
                if student_name == 'config':
                    continue
                
                posttest_results[teacher_name][student_name]['config'] = student_data['config']
                student_config = student_data['config']
                api_key, base_url = self._get_student_api_info(student_name, student_config['model'])
                student = StudentLLM(**student_config, api_key=api_key, base_url=base_url)
                
                with ThreadPoolExecutor(max_workers=CONFIG["PARALLEL_TASKS"]) as executor:
                    future_to_question = {}
                    
                    for question_id, question_data in student_data.items():
                        if question_id == 'config':
                            continue
                        
                        category = question_data['category']
                        interaction_history = question_data['interaction']
                        pre_test_result = question_data['pre_test']['responses']
                        few_shot_cot_examples = self.val_data.get(category, self.val_data.get("general", []))

                        future = executor.submit(
                            student.take_test,
                            pre_test_result,
                            few_shot_cot_examples,
                            interaction_history,
                            pre_test_result
                        )
                        future_to_question[future] = (teacher_name, student_name, question_id, category, question_data)

                    for future in tqdm(
                        as_completed(future_to_question),
                        total=len(future_to_question),
                        desc=f"Posttest for {student_name} with {teacher_name}"
                    ):
                        teacher_name, student_name, question_id, category, question_data = future_to_question[future]
                        try:
                            post_test_results = future.result()
                            post_test_scores = self.evaluator.calculate_accuracy(post_test_results)

                            posttest_results[teacher_name][student_name][question_id] = {
                                "category": category,
                                "pre_test": question_data['pre_test'],
                                "interaction": question_data['interaction'],
                                "post_test": {
                                    "responses": post_test_results,
                                    "scores": post_test_scores,
                                }
                            }
                        except Exception as e:
                            logging.error(f"Error in posttest for question {question_id} with teacher {teacher_name} and student {student_name}: {str(e)}")

        self._save_results(dict(posttest_results), "pretest_interaction_posttest_results")
        return dict(posttest_results)
    
    def _run_posttest_from_json(self, interaction_results_path: str):
        with open(interaction_results_path, "r") as f:
            interaction_results = json.load(f)
        return self._run_posttest(interaction_results)
    
    def _perform_evaluation(self, results):
        evaluation = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for teacher_name, teacher_data in results.items():
            if teacher_name == 'config':
                continue
            for student_name, student_data in teacher_data.items():
                if student_name == 'config':
                    continue
                category_results = defaultdict(lambda: {"pre_test_correct": 0, "post_test_correct": 0, "total": 0})
                
                for question_id, question_data in student_data.items():
                    if question_id == 'config':
                        continue
                    category = question_data['category']
                    pre_test = question_data['pre_test']['responses'][0]
                    post_test = question_data['post_test']['responses'][0]

                    category_results[category]['total'] += 1
                    if pre_test['model_prediction'] == pre_test['correct_answer']:
                        category_results[category]['pre_test_correct'] += 1
                    if post_test['model_prediction'] == post_test['correct_answer']:
                        category_results[category]['post_test_correct'] += 1

                for category, results in category_results.items():
                    total = results['total']
                    pre_test_accuracy = results['pre_test_correct'] / total
                    post_test_accuracy = results['post_test_correct'] / total
                    progress = post_test_accuracy - pre_test_accuracy

                    evaluation[teacher_name][student_name][category] = {
                        "pre_test_accuracy": pre_test_accuracy,
                        "post_test_accuracy": post_test_accuracy,
                        "progress": progress,
                    }

                # Calculate overall results
                total_pre_test_correct = sum(r['pre_test_correct'] for r in category_results.values())
                total_post_test_correct = sum(r['post_test_correct'] for r in category_results.values())
                total_questions = sum(r['total'] for r in category_results.values())

                evaluation[teacher_name][student_name]['overall'] = {
                    "pre_test_accuracy": total_pre_test_correct / total_questions,
                    "post_test_accuracy": total_post_test_correct / total_questions,
                    "progress": (total_post_test_correct - total_pre_test_correct) / total_questions,
                }

                # Add student config to evaluation results
                evaluation[teacher_name][student_name]['config'] = student_data['config']

            # Add teacher config to evaluation results
            evaluation[teacher_name]['config'] = teacher_data['config']


        self._save_results(dict(evaluation), "evaluation_results")
        return dict(evaluation)

    def _interaction_evaluation(self, posttest_results_path: str, csv_path: str):
        # Load posttest results
        with open(posttest_results_path, 'r', encoding='utf-8') as f:
            posttest_results = json.load(f)

        # Load CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            evaluation_tasks = list(csv_reader)

        evaluation_results = {}

        for question_id, teacher_a_name, teacher_b_name in evaluation_tasks:
            if question_id not in evaluation_results:
                evaluation_results[question_id] = {}

            # Prepare data for evaluation
            category = posttest_results[teacher_a_name][self.config.student_configs[0]['name']][question_id]['category']
            pre_test_result = posttest_results[teacher_a_name][self.config.student_configs[0]['name']][question_id]['pre_test']
            teacher_a_interaction = posttest_results[teacher_a_name][self.config.student_configs[0]['name']][question_id]['interaction']
            teacher_b_interaction = posttest_results[teacher_b_name][self.config.student_configs[0]['name']][question_id]['interaction']

            # Perform evaluation
            evaluation = self.evaluator.over_interaction_analysis(
                question_id,
                category,
                pre_test_result,
                teacher_a_name,
                teacher_a_interaction,
                teacher_b_name,
                teacher_b_interaction
            )

            evaluation_results[question_id][f"{teacher_a_name}_vs_{teacher_b_name}"] = evaluation

        # Save evaluation results
        self._save_results(evaluation_results, "interaction_evaluation_results")

        logging.info(f"Interaction evaluation completed. Results saved.")
        return evaluation_results

    def _teacher_questions_evaluation(self, posttest_results_path: str, csv_path: str):
        # Load posttest results
        with open(posttest_results_path, 'r', encoding='utf-8') as f:
            posttest_results = json.load(f)

        # Load CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            evaluation_tasks = list(csv_reader)

        evaluation_results = {}

        for question_id, teacher_1_name, teacher_2_name in evaluation_tasks:
            if question_id not in evaluation_results:
                evaluation_results[question_id] = {}

            # Prepare data for evaluation
            category = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['category']
            teacher_1_interaction = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['interaction']
            teacher_2_interaction = posttest_results[teacher_2_name][self.config.student_configs[0]['name']][question_id]['interaction']

            # Perform evaluation
            evaluation = self.evaluator.teacher_questions_analysis(
                question_id,
                category,
                teacher_1_name,
                teacher_1_interaction,
                teacher_2_name,
                teacher_2_interaction
            )

            evaluation_results[question_id][f"{teacher_1_name}_vs_{teacher_2_name}"] = evaluation

        # Save evaluation results
        self._save_results(evaluation_results, "teacher_questions_evaluation_results")

        logging.info(f"Teacher questions evaluation completed. Results saved.")
        return evaluation_results
    
    def _student_responses_evaluation(self, posttest_results_path: str, csv_path: str):
        # Load posttest results
        with open(posttest_results_path, 'r', encoding='utf-8') as f:
            posttest_results = json.load(f)

        # Load CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            evaluation_tasks = list(csv_reader)

        evaluation_results = {}

        for question_id, teacher_1_name, teacher_2_name in evaluation_tasks:
            if question_id not in evaluation_results:
                evaluation_results[question_id] = {}

            # Prepare data for evaluation
            category = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['category']
            teacher_1_interaction = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['interaction']
            teacher_2_interaction = posttest_results[teacher_2_name][self.config.student_configs[0]['name']][question_id]['interaction']

            # Perform evaluation
            evaluation = self.evaluator.student_responses_analysis(
                question_id,
                category,
                teacher_1_name,
                teacher_1_interaction,
                teacher_2_name,
                teacher_2_interaction
            )

            evaluation_results[question_id][f"{teacher_1_name}_vs_{teacher_2_name}"] = evaluation

        # Save evaluation results
        self._save_results(evaluation_results, "student_responses_within_teacher_evaluation_results")

        logging.info(f"Student responses within teacher evaluation completed. Results saved.")
        return evaluation_results

    def _comprehensive_evaluation(self, posttest_results_path: str, csv_path: str):
        # Load posttest results
        with open(posttest_results_path, 'r', encoding='utf-8') as f:
            posttest_results = json.load(f)

        # Load CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            evaluation_tasks = list(csv_reader)

        comprehensive_results = {}

        for question_id, teacher_1_name, teacher_2_name in evaluation_tasks:
            if question_id not in comprehensive_results:
                comprehensive_results[question_id] = {}

            pair_key = f"{teacher_1_name}_vs_{teacher_2_name}"
            if pair_key not in comprehensive_results[question_id]:
                comprehensive_results[question_id][pair_key] = {}

            # Prepare data for evaluation
            category = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['category']
            pre_test_result = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['pre_test']
            teacher_1_interaction = posttest_results[teacher_1_name][self.config.student_configs[0]['name']][question_id]['interaction']
            teacher_2_interaction = posttest_results[teacher_2_name][self.config.student_configs[0]['name']][question_id]['interaction']

            # Perform interaction evaluation
            interaction_evaluation = self.evaluator.over_interaction_analysis(
                question_id,
                category,
                pre_test_result,
                teacher_1_name,
                teacher_1_interaction,
                teacher_2_name,
                teacher_2_interaction
            )
            comprehensive_results[question_id][pair_key]['interaction_analysis'] = interaction_evaluation

            # Perform teacher questions evaluation
            teacher_questions_evaluation = self.evaluator.teacher_questions_analysis(
                question_id,
                category,
                teacher_1_name,
                teacher_1_interaction,
                teacher_2_name,
                teacher_2_interaction
            )
            comprehensive_results[question_id][pair_key]['teacher_questions_analysis'] = teacher_questions_evaluation

            # Perform student responses evaluation
            student_responses_evaluation = self.evaluator.student_responses_analysis(
                question_id,
                category,
                teacher_1_name,
                teacher_1_interaction,
                teacher_2_name,
                teacher_2_interaction
            )
            comprehensive_results[question_id][pair_key]['student_responses_analysis'] = student_responses_evaluation

        # Save comprehensive evaluation results
        self._save_results(comprehensive_results, "comprehensive_evaluation_results")

        logging.info(f"Comprehensive evaluation completed. Results saved.")
        return comprehensive_results

    def _save_results(self, results, prefix):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.config.experiment_version}_{timestamp}.json"
        with open(os.path.join(self.config.output_path, filename), "w") as f:
            json.dump(results, f, indent=2)


def main():
    config = EvalConfig.from_yaml(CONFIG["CONFIG_YAML_FILEPATH"])
    os.makedirs(config.output_path, exist_ok=True)
    setup_logging(config.logging_level, config.output_path)
    manager = EvalManager(config)
    
    try:
        # results = manager.run_complete_eval()

        pretest_results = manager._run_pretest()

        pretest_interaction_results = manager._run_interactions(pretest_results)
        # pretest_interaction_results = manager._run_interaction_from_json(config.output_path + "GPQA-diamond_Student-llama31-70b-instruct_pretest/GPQA-diamond_Student-llama31-70b-instruct_zero-shot_pretest-results_1.0.0_20240928_162052.json")
        # pretest_interaction_results = manager._run_interaction_from_json(config.output_path + "GPQA-main_Student-llama31-70b-instruct_pretest/GPQA-diamond_Student-llama31-70b-instruct_zero-shot_pretest-results_1.0.0_20240905_000343.json")
        # pretest_interaction_results = manager._run_interaction_from_json(config.output_path + "MMLU-Pro-strastified_Student-llama31-70b-instruct_pretest/MMLU-Pro-stratified_Student-llama31-70b-instruct_zero-shot_pretest-results_1.0.0_212612.json")
        # pretest_interaction_results = manager._run_interaction_from_json(config.output_path + "pretest_results_1.0.0_20241018_170901.json")
        # pretest_interaction_results = manager._run_interaction_from_json(config.output_path + "MMLU-Pro-stratified_Student-llama31-70b-instruct_pretest/MMLU-Pro-stratified_Student-llama31-70b-instruct_zero-shot_pretest-results_1.0.0_212612.json")

        posttest_results = manager._run_posttest(pretest_interaction_results)
        # posttest_results = manager._run_posttest_from_json(config.output_path + "GPQA-diamond_13-Teachers_Student-llama31-70b-instruct/13teachers_pretest_interaction_results_1.0.0.json")
        # posttest_results = manager._run_posttest_from_json(config.output_path + "pretest_interaction_results_1.0.0_20240928_175620.json")
        # posttest_results = manager._run_posttest_from_json(config.output_path + "pretest_interaction_results_1.0.0_20241007_174145.json")
        
        evaluation_results = manager._perform_evaluation(posttest_results)
        
        # posttest_results_path = "D:/Workspace/EducationQ_Benchmark/src/data/output/EduQ-Bench_Student-llama31-70b-instruct/MMLU-Pro/data/MMLU-Pro-stratified/Student-llama31-70b-instruct/processed_results/merged_results_rerun_empty_interaction_posttest_3_rerun_checked_posttest_3.json"
        # csv_path = "D:/Workspace/EducationQ_Benchmark/src/data/output/EduQ-Bench_Student-llama31-70b-instruct/MMLU-Pro/data/JudgeLLM_list_20241011_2.csv"
        
        # evaluation_results = manager._interaction_evaluation(posttest_results_path, csv_path)
        # evaluation_results = manager._teacher_questions_evaluation(posttest_results_path, csv_path)
        # evaluation_results = manager._student_responses_evaluation(posttest_results_path, csv_path)

        #evaluation_results = manager._comprehensive_evaluation(posttest_results_path, csv_path)

        logging.info(f"Interaction evaluation completed successfully.")
        
        # logging.info(f"Experiment {config.experiment_version} completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during the experiment: {str(e)}")
        raise

if __name__ == "__main__":
    main()