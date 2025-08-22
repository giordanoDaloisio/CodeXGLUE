#!/usr/bin/env python3
"""
Compare Teacher vs Student Model Performance
Evaluate both LLaMA-3.1-8B (teacher) and distilled LLaMA-3.2-1B (student) on the same test set
"""

import json
import os
import time
import argparse
from typing import List, Dict, Any
import torch
from evaluate_distilled_model import DistilledModelEvaluator
from code_summarization_llama import CodeSummarizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare performance between teacher and student models"""
    
    def __init__(self, student_model_path: str, teacher_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.student_model_path = student_model_path
        self.teacher_model_name = teacher_model_name
        
        # Initialize student evaluator
        logger.info("Loading student (distilled) model...")
        self.student_evaluator = DistilledModelEvaluator(student_model_path)
        
        # Initialize teacher model
        logger.info("Loading teacher model...")
        
        # Create a minimal args object for teacher model
        class TeacherArgs:
            def __init__(self):
                self.quantf8 = False
                self.quanti4 = False
                self.quanti8 = False
                self.prune20 = False
                self.prune40 = False
                self.prune60 = False
                self.gradual_pruning = False
                self.offloaded_pruning = False
                self.cpu_only_pruning = False
        
        teacher_args = TeacherArgs()
        self.teacher_model = CodeSummarizer(
            model_name=teacher_model_name,
            job_id="comparison",
            args=teacher_args
        )
        
        logger.info("Both models loaded successfully")
    
    def get_model_stats(self, model):
        """Get model statistics"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1e6
        total_params = sum(p.numel() for p in model.parameters())
        return size_mb, total_params
    
    def compare_models(self, validation_file: str, test_file: str, 
                      num_few_shot: int = 3, num_test_samples: int = 50,
                      output_dir: str = "./comparison_results"):
        """
        Compare teacher and student models on the same test set
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data (same for both models)
        logger.info("Loading test data...")
        test_data = self.student_evaluator.load_jsonl(test_file)
        validation_data = self.student_evaluator.load_jsonl(validation_file)
        
        # Sample test data
        if num_test_samples and num_test_samples < len(test_data):
            import random
            random.seed(42)  # For reproducible comparison
            test_data = random.sample(test_data, num_test_samples)
            logger.info(f"Using {num_test_samples} test samples for comparison")
        
        # Select few-shot examples (same for both models)
        import random
        random.seed(42)
        few_shot_examples = random.sample(validation_data, min(num_few_shot, len(validation_data)))
        
        # Get model statistics
        student_size, student_params = self.get_model_stats(self.student_evaluator.model)
        teacher_size, teacher_params = self.get_model_stats(self.teacher_model.model)
        
        logger.info(f"Teacher Model: {teacher_size:.2f} MB, {teacher_params:,} parameters")
        logger.info(f"Student Model: {student_size:.2f} MB, {student_params:,} parameters")
        logger.info(f"Compression Ratio: {teacher_size/student_size:.2f}x smaller, {teacher_params/student_params:.2f}x fewer parameters")
        
        # Evaluate student model
        logger.info("\n" + "="*50)
        logger.info("EVALUATING STUDENT MODEL (Distilled LLaMA-3.2-1B)")
        logger.info("="*50)
        
        student_results = []
        student_times = []
        
        for i, test_example in enumerate(test_data):
            prompt = self.student_evaluator.create_few_shot_prompt(few_shot_examples, test_example['code'])
            
            start_time = time.time()
            student_summary = self.student_evaluator.generate_summary(prompt)
            end_time = time.time()
            
            student_times.append(end_time - start_time)
            student_results.append({
                'repo': test_example['repo'],
                'path': test_example['path'],
                'func_name': test_example['func_name'],
                'original_code': test_example['code'],
                'original_summary': test_example['docstring'],
                'generated_summary': student_summary,
                'inference_time': end_time - start_time
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Student: Processed {i+1}/{len(test_data)} samples")
        
        # Evaluate teacher model
        logger.info("\n" + "="*50)
        logger.info("EVALUATING TEACHER MODEL (LLaMA-3.1-8B)")
        logger.info("="*50)
        
        teacher_results = []
        teacher_times = []
        
        for i, test_example in enumerate(test_data):
            prompt = self.teacher_model.create_few_shot_prompt(few_shot_examples, test_example['code'])
            
            start_time = time.time()
            teacher_summary = self.teacher_model.generate_summary(prompt)
            end_time = time.time()
            
            teacher_times.append(end_time - start_time)
            teacher_results.append({
                'repo': test_example['repo'],
                'path': test_example['path'], 
                'func_name': test_example['func_name'],
                'original_code': test_example['code'],
                'original_summary': test_example['docstring'],
                'generated_summary': teacher_summary,
                'inference_time': end_time - start_time
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Teacher: Processed {i+1}/{len(test_data)} samples")
        
        # Create comparison results
        comparison_results = []
        for i in range(len(test_data)):
            comparison_results.append({
                'sample_id': i,
                'repo': test_data[i]['repo'],
                'path': test_data[i]['path'],
                'func_name': test_data[i]['func_name'],
                'original_code': test_data[i]['code'],
                'ground_truth_summary': test_data[i]['docstring'],
                'teacher_summary': teacher_results[i]['generated_summary'],
                'student_summary': student_results[i]['generated_summary'],
                'teacher_time': teacher_results[i]['inference_time'],
                'student_time': student_results[i]['inference_time'],
                'speedup': teacher_results[i]['inference_time'] / student_results[i]['inference_time']
            })
        
        # Calculate statistics
        avg_teacher_time = sum(teacher_times) / len(teacher_times)
        avg_student_time = sum(student_times) / len(student_times)
        avg_speedup = avg_teacher_time / avg_student_time
        
        total_teacher_time = sum(teacher_times)
        total_student_time = sum(student_times)
        
        # Compilation statistics
        stats = {
            'comparison_metadata': {
                'teacher_model': self.teacher_model_name,
                'student_model_path': self.student_model_path,
                'num_test_samples': len(test_data),
                'num_few_shot_examples': len(few_shot_examples),
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'model_sizes': {
                'teacher_size_mb': teacher_size,
                'teacher_parameters': teacher_params,
                'student_size_mb': student_size,
                'student_parameters': student_params,
                'size_compression_ratio': teacher_size / student_size,
                'parameter_compression_ratio': teacher_params / student_params
            },
            'performance_metrics': {
                'avg_teacher_inference_time': avg_teacher_time,
                'avg_student_inference_time': avg_student_time,
                'average_speedup': avg_speedup,
                'total_teacher_time': total_teacher_time,
                'total_student_time': total_student_time,
                'total_speedup': total_teacher_time / total_student_time
            }
        }
        
        # Save results
        comparison_file = os.path.join(output_dir, "teacher_vs_student_comparison.json")
        teacher_file = os.path.join(output_dir, "teacher_results.json")
        student_file = os.path.join(output_dir, "student_results.json")
        stats_file = os.path.join(output_dir, "comparison_stats.json")
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        with open(teacher_file, 'w', encoding='utf-8') as f:
            json.dump(teacher_results, f, indent=2, ensure_ascii=False)
        
        with open(student_file, 'w', encoding='utf-8') as f:
            json.dump(student_results, f, indent=2, ensure_ascii=False)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"📊 Model Sizes:")
        logger.info(f"   Teacher: {teacher_size:.2f} MB ({teacher_params:,} parameters)")
        logger.info(f"   Student: {student_size:.2f} MB ({student_params:,} parameters)")
        logger.info(f"   Compression: {teacher_size/student_size:.2f}x smaller, {teacher_params/student_params:.2f}x fewer parameters")
        logger.info(f"")
        logger.info(f"⚡ Performance:")
        logger.info(f"   Teacher avg time: {avg_teacher_time:.3f} seconds/sample")
        logger.info(f"   Student avg time: {avg_student_time:.3f} seconds/sample")
        logger.info(f"   Speedup: {avg_speedup:.2f}x faster")
        logger.info(f"")
        logger.info(f"📁 Results saved to:")
        logger.info(f"   Comparison: {comparison_file}")
        logger.info(f"   Statistics: {stats_file}")
        logger.info(f"   Teacher results: {teacher_file}")
        logger.info(f"   Student results: {student_file}")
        
        # Show some example comparisons
        logger.info(f"\n📋 Sample Comparisons:")
        logger.info("-" * 40)
        for i in range(min(3, len(comparison_results))):
            result = comparison_results[i]
            logger.info(f"\nExample {i+1}: {result['func_name']}")
            logger.info(f"Ground Truth: {result['ground_truth_summary']}")
            logger.info(f"Teacher:      {result['teacher_summary']}")
            logger.info(f"Student:      {result['student_summary']}")
            logger.info(f"Times:        Teacher {result['teacher_time']:.3f}s, Student {result['student_time']:.3f}s (speedup: {result['speedup']:.2f}x)")
        
        return comparison_results, stats

def main():
    parser = argparse.ArgumentParser(description="Compare Teacher vs Student Model Performance")
    
    parser.add_argument("--student_model_path", required=True,
                       help="Path to the distilled (student) model directory")
    parser.add_argument("--teacher_model", 
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Teacher model name")
    parser.add_argument("--validation_file", 
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl",
                       help="Path to validation JSONL file")
    parser.add_argument("--test_file",
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/test.jsonl", 
                       help="Path to test JSONL file")
    parser.add_argument("--num_few_shot", type=int, default=3,
                       help="Number of few-shot examples to use")
    parser.add_argument("--num_test_samples", type=int, default=50,
                       help="Number of test samples to evaluate")
    parser.add_argument("--output_dir", default="./comparison_results",
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Initialize comparator
    logger.info("Initializing Model Comparator...")
    comparator = ModelComparator(
        student_model_path=args.student_model_path,
        teacher_model_name=args.teacher_model
    )
    
    # Run comparison
    comparison_results, stats = comparator.compare_models(
        validation_file=args.validation_file,
        test_file=args.test_file,
        num_few_shot=args.num_few_shot,
        num_test_samples=args.num_test_samples,
        output_dir=args.output_dir
    )
    
    logger.info("Comparison completed successfully!")

if __name__ == "__main__":
    main()
