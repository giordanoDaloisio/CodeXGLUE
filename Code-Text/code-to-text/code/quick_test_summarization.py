#!/usr/bin/env python3
"""
Quick test script for code summarization with a few examples
"""

import json
import random
from code_summarization_llama import CodeSummarizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Run a quick test with just a few examples"""
    
    # Initialize summarizer
    logger.info("Initializing Code Summarizer...")
    summarizer = CodeSummarizer()
    
    # Load a few examples from validation and test
    validation_file = "/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl"
    test_file = "/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/test.jsonl"
    
    validation_data = summarizer.load_jsonl(validation_file)
    test_data = summarizer.load_jsonl(test_file)
    
    # Select random examples
    few_shot_examples = random.sample(validation_data, 3)
    test_examples = random.sample(test_data, 5)  # Just 5 for quick test
    
    logger.info("Few-shot examples:")
    for i, ex in enumerate(few_shot_examples, 1):
        logger.info(f"{i}. {ex['func_name']}: {ex['docstring'][:100]}...")
    
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    
    for i, test_example in enumerate(test_examples, 1):
        print(f"\nTest Example {i}:")
        print(f"Function: {test_example['func_name']}")
        print(f"Repository: {test_example['repo']}")
        
        # Create prompt and generate summary
        prompt = summarizer.create_few_shot_prompt(few_shot_examples, test_example['code'])
        generated_summary = summarizer.generate_summary(prompt, temperature=0.1)
        
        print(f"\nOriginal Summary:")
        print(f"  {test_example['docstring']}")
        print(f"\nGenerated Summary:")
        print(f"  {generated_summary}")
        print("-" * 80)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    quick_test()
