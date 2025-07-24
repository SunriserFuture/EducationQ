import json
import os
from collections import defaultdict
from typing import Dict, List, Any
import tiktoken
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from educationq_benchmark_v3_2 import EvalManager, StudentLLM, TeacherLLM, EvaluatorLLM, EvalConfig, CONFIG

def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def process_results(file_path: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    empty_responses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for teacher_name, teacher_data in results.items():
        for student_name, student_data in teacher_data.items():
            if student_name == 'config':
                continue
            for question_id, question_data in student_data.items():
                if question_id == 'config':
                    continue
                
                post_test = question_data.get('post_test', {}).get('responses', [])
                if post_test:
                    tokens = count_tokens(post_test[0].get('model_response', ''))
                    if tokens == 0:
                        empty_responses[teacher_name][student_name]['rerun_posttest'].append(question_id)
                    elif tokens > 1638:
                        empty_responses[teacher_name][student_name]['rerun_interactions_and_posttest'].append(question_id)

    return empty_responses

def print_empty_responses(empty_responses: Dict[str, Dict[str, Dict[str, List[str]]]]):
    for teacher, students in empty_responses.items():
        print(f"Teacher: {teacher}")
        for student, rerun_types in students.items():
            print(f"  Student: {student}")
            for rerun_type, questions in rerun_types.items():
                print(f"    {rerun_type}: {len(questions)}")
                for question in questions:
                    print(f"      {question}")

def select_pairs_to_rerun(empty_responses: Dict[str, Dict[str, Dict[str, List[str]]]]):
    pairs = []
    all_pairs = []
    
    print("Available teacher-student pairs:")
    for i, (teacher, students) in enumerate(empty_responses.items(), 1):
        for j, student in enumerate(students.keys(), 1):
            pair_index = len(all_pairs) + 1
            all_pairs.append((teacher, student))
            print(f"{pair_index}. Teacher: {teacher}, Student: {student}")
    
    while True:
        selection = input("Enter the number of the pair to rerun (or press Enter to finish): ").strip()
        if not selection:
            break
        try:
            index = int(selection) - 1
            if 0 <= index < len(all_pairs):
                pairs.append(all_pairs[index])
                print(f"Selected: Teacher: {all_pairs[index][0]}, Student: {all_pairs[index][1]}")
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    return pairs

def rerun_posttest(
    eval_manager: EvalManager,
    results: Dict[str, Any],
    pairs_to_rerun: List[tuple],
    empty_responses: Dict[str, Dict[str, Dict[str, List[str]]]]
) -> Dict[str, Any]:
    for teacher_name, student_name in pairs_to_rerun:
        teacher_config = results[teacher_name]['config']
        student_config = results[teacher_name][student_name]['config']
        
        student_api_key, student_base_url = eval_manager._get_student_api_info(student_name, student_config['model'])
        student = StudentLLM(**student_config, api_key=student_api_key, base_url=student_base_url)

        questions_to_rerun_posttest = empty_responses[teacher_name][student_name]['rerun_posttest']
        questions_to_rerun_interactions = empty_responses[teacher_name][student_name]['rerun_interactions_and_posttest']
        
        # Rerun posttest only
        for question_id in questions_to_rerun_posttest:
            question_data = results[teacher_name][student_name][question_id]
            category = question_data['category']
            pre_test_result = question_data['pre_test']['responses']
            interaction_history = question_data['interaction']
            few_shot_cot_examples = eval_manager.val_data.get(category, eval_manager.val_data.get("general", []))

            post_test_results = student.take_test(
                pre_test_result,
                few_shot_cot_examples,
                interaction_history,
                pre_test_result
            )
            post_test_scores = eval_manager.evaluator.calculate_accuracy(post_test_results)

            # 更新结果
            results[teacher_name][student_name][question_id]['post_test'] = {
                "responses": post_test_results,
                "scores": post_test_scores,
            }

        # Rerun interactions and posttest
        teacher_api_key, teacher_base_url = eval_manager._get_teacher_api_info(teacher_name, teacher_config['model'])
        teacher = TeacherLLM(**teacher_config, api_key=teacher_api_key, base_url=teacher_base_url)

        for question_id in questions_to_rerun_interactions:
            question_data = results[teacher_name][student_name][question_id]
            category = question_data['category']
            pre_test_result = question_data['pre_test']['responses']
            few_shot_cot_examples = eval_manager.val_data.get(category, eval_manager.val_data.get("general", []))

            # Rerun interactions
            interaction_history = []
            for i in range(eval_manager.config.num_interactions):
                teacher_question = teacher.generate_question(
                    category,
                    pre_test_result,
                    interaction_history,
                    i + 1,
                    eval_manager.config.num_interactions,
                    few_shot_cot_examples if teacher.use_few_shot else None,
                )
                student_answer = student.answer_question(
                    category, 
                    teacher_question, 
                    interaction_history, 
                    pre_test_result, 
                    few_shot_cot_examples if student.use_few_shot else None,
                )
                interaction_history.append({"question": teacher_question, "answer": student_answer})

            # Rerun posttest
            post_test_results = student.take_test(
                pre_test_result,
                few_shot_cot_examples,
                interaction_history,
                pre_test_result
            )
            post_test_scores = eval_manager.evaluator.calculate_accuracy(post_test_results)

            # 更新结果
            results[teacher_name][student_name][question_id]['interaction'] = interaction_history
            results[teacher_name][student_name][question_id]['post_test'] = {
                "responses": post_test_results,
                "scores": post_test_scores,
            }

    return results

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 初始化EvalManager
    config_path = "D:/Workspace/EducationQ_Benchmark/src/data/input/config_teacher0shot_gpqa_diamond_0928.yaml"  # 替换为你的配置文件路径
    config = EvalConfig.from_yaml(config_path)
    eval_manager = EvalManager(config)

    # 处理结果文件
    results_file = "D:/Workspace/EducationQ_Benchmark/src/data/output/EduQ-Bench_Student-llama31-70b-instruct/GPQA/data/GPQA-diamond/Student-llama31-70b-instruct/processed_results/merged_results_1_rerun_interaction_rerun_posttest_3.json"  # 替换为你的结果文件路径
    empty_responses = process_results(results_file)

    # 打印空响应统计
    print_empty_responses(empty_responses)

    # 选择要重新运行的师生对
    pairs_to_rerun = select_pairs_to_rerun(empty_responses)

    if pairs_to_rerun:
        # 加载原始结果
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 重新运行posttest和interactions
        updated_results = rerun_posttest(eval_manager, results, pairs_to_rerun, empty_responses)

        # 保存更新后的结果
        output_file = "D:/Workspace/EducationQ_Benchmark/src/data/output/EduQ-Bench_Student-llama31-70b-instruct/GPQA/data/GPQA-diamond/Student-llama31-70b-instruct/processed_results/merged_results_1_rerun_interaction_rerun_posttest_4.json"  # 替换为你想要保存的新文件路径
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_results, f, indent=2, ensure_ascii=False)

        print(f"Updated results saved to {output_file}")
    else:
        print("No pairs selected for rerun. Exiting.")

if __name__ == "__main__":
    main()