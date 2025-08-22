#!/usr/bin/env python3
"""
Evaluation script for distilled LLaMA model on code summarization
"""

import json
import os
import random
import time
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistilledModelEvaluator:
    """Evaluator for the distilled model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading distilled model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        logger.info("Distilled model loaded successfully")
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            logger.info(f"Loaded {len(data)} examples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
        return data
    
    def create_few_shot_prompt(self, few_shot_examples: List[Dict[str, Any]], test_code: str) -> str:
        """Create a few-shot prompt for code summarization"""
        prompt = """You are an expert software developer tasked with writing concise and accurate summaries for Java methods. Given a Java method, provide a clear, one-sentence summary that describes what the method does.

Here are some examples:

"""
        
        # Add few-shot examples
        for i, example in enumerate(few_shot_examples, 1):
            code = example['code'].strip()
            summary = example['docstring'].strip()
            
            prompt += f"Example {i}:\nCode:\n```java\n{code}\n```\nSummary: {summary}\n\n"
        
        # Add the test case
        prompt += f"Now, please provide a summary for this Java method:\nCode:\n```java\n{test_code.strip()}\n```\nSummary:"
        
        return prompt
    
    def generate_summary(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.1) -> str:
        """Generate summary using the distilled model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1000,  # Leave room for generation
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after "Summary:")
            if "Summary:" in full_response:
                summary = full_response.split("Summary:")[-1].strip()
                # Take only the first line/sentence
                summary = summary.split('\n')[0].strip()
                # Remove any markdown or extra formatting
                summary = summary.replace('```', '').strip()
                return summary
            else:
                return "Error: Could not extract summary"
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error: {str(e)}"
    
    def print_model_size(self):
        """Calculate and print model size"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1e6
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Model size: {size_mb:.2f} MB")
        logger.info(f"Total parameters: {total_params:,}")
        return size_mb, total_params
    
    def evaluate(self, validation_file: str, test_file: str, 
                 num_few_shot: int = 3, num_test_samples: int = 100, 
                 output_file: str = None, job_id: str = None):
        """Evaluate the distilled model"""
        
        # Load data
        logger.info("Loading datasets...")
        validation_data = self.load_jsonl(validation_file)
        test_data = self.load_jsonl(test_file)
        
        # Sample test data if specified
        if num_test_samples and num_test_samples < len(test_data):
            test_data = random.sample(test_data, num_test_samples)
            logger.info(f"Using {num_test_samples} test samples")
        
        # Select few-shot examples
        few_shot_examples = random.sample(validation_data, min(num_few_shot, len(validation_data)))
        logger.info(f"Using {len(few_shot_examples)} few-shot examples")
        
        # Print model statistics
        model_size, total_params = self.print_model_size()
        
        # GPU Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(5):  # Warmup for 5 iterations
                inputs = self.tokenizer("print('Hello, world!')", return_tensors="pt").to(self.device)
                _ = self.model(**inputs)
        
        results = []
        times = []
        start_time = time.time()
        
        # Process test samples
        logger.info("Starting evaluation...")
        for i, test_example in enumerate(tqdm(test_data, desc="Generating summaries")):
            try:
                # Create prompt
                prompt = self.create_few_shot_prompt(few_shot_examples, test_example['code'])
                
                # Generate summary
                start = time.time()
                generated_summary = self.generate_summary(prompt)
                end = time.time()
                times.append(end - start)
                
                # Store result
                result = {
                    'repo': test_example['repo'],
                    'path': test_example['path'],
                    'func_name': test_example['func_name'],
                    'original_code': test_example['code'],
                    'original_summary': test_example['docstring'],
                    'generated_summary': generated_summary,
                    'url': test_example.get('url', ''),
                }
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (len(test_data) - i - 1)
                    logger.info(f"Processed {i+1}/{len(test_data)} samples. Est. remaining time: {remaining/60:.1f}min")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                result = {
                    'repo': test_example['repo'],
                    'path': test_example['path'],
                    'func_name': test_example['func_name'],
                    'original_code': test_example['code'],
                    'original_summary': test_example['docstring'],
                    'generated_summary': f"Error: {str(e)}",
                    'url': test_example.get('url', ''),
                }
                results.append(result)
        
        # Save results
        if output_file is None:
            output_file = f"distilled_model_results_{job_id if job_id else 'test'}.json"
            time_file = f"distilled_model_times_{job_id if job_id else 'test'}.csv"
        else:
            time_file = output_file.replace('.json', '_times.csv')
        
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save timing information
        with open(time_file, "w", encoding="utf-8") as f:
            json.dump(times, f, indent=2, ensure_ascii=False)
        
        # Print some example results
        logger.info("\n" + "="*50)
        logger.info("SAMPLE RESULTS:")
        logger.info("="*50)
        for i, result in enumerate(results[:3]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Function: {result['func_name']}")
            logger.info(f"Original: {result['original_summary']}")
            logger.info(f"Generated: {result['generated_summary']}")
            logger.info("-" * 30)
        
        total_time = time.time() - start_time
        avg_inference_time = sum(times) / len(times) if times else 0
        
        logger.info(f"\nEvaluation completed in {total_time/60:.1f} minutes")
        logger.info(f"Average inference time per sample: {avg_inference_time:.3f} seconds")
        logger.info(f"Model size: {model_size:.2f} MB ({total_params:,} parameters)")
        logger.info(f"Results saved to: {output_file}")
        
        # Save summary statistics
        stats = {
            'model_path': self.model_path,
            'model_size_mb': model_size,
            'total_parameters': total_params,
            'num_test_samples': len(results),
            'total_evaluation_time_minutes': total_time / 60,
            'average_inference_time_seconds': avg_inference_time,
            'output_file': output_file,
            'time_file': time_file
        }
        
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return results, stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate Distilled LLaMA Model for Code Summarization")
    
    parser.add_argument("--model_path", required=True,
                       help="Path to the distilled model directory")
    parser.add_argument("--validation_file", 
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl",
                       help="Path to validation JSONL file")
    parser.add_argument("--test_file",
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/test.jsonl", 
                       help="Path to test JSONL file")
    parser.add_argument("--num_few_shot", type=int, default=3,
                       help="Number of few-shot examples to use")
    parser.add_argument("--num_test_samples", type=int, default=100,
                       help="Number of test samples to evaluate")
    parser.add_argument("--output_file", default=None,
                       help="Output file path for results")
    parser.add_argument("--job_id", type=str, default=None,
                       help="Job ID for output file naming")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature for generation")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Initialize evaluator
    logger.info("Initializing Distilled Model Evaluator...")
    evaluator = DistilledModelEvaluator(model_path=args.model_path)
    
    # Run evaluation
    results, stats = evaluator.evaluate(
        validation_file=args.validation_file,
        test_file=args.test_file,
        num_few_shot=args.num_few_shot,
        num_test_samples=args.num_test_samples,
        output_file=args.output_file,
        job_id=args.job_id
    )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Total examples processed: {len(results)}")
    logger.info(f"Model size: {stats['model_size_mb']:.2f} MB")
    logger.info(f"Average inference time: {stats['average_inference_time_seconds']:.3f} seconds")

if __name__ == "__main__":
    main()
