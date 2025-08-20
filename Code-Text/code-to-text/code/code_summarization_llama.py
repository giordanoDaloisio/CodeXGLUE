#!/usr/bin/env python3
"""
Code Summarization using LLaMA-3.1-8B Instruct with Few-shot Prompting
Uses CodeXGLUE Java dataset for validation (few-shot) and testing
"""

import json
import os
import random
import time
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
import logging
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeSummarizer:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", job_id=None, args=None):
        """
        Initialize the code summarizer with LLaMA model
        
        Args:
            model_name: HuggingFace model name
            use_quantization: Whether to use 8-bit quantization to reduce memory usage
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.job_id = job_id
        self.args = args
        logger.info(f"Using device: {self.device}")
        
        # Configure quantization if requested
        quantization_config = None
        if args.quantf8 or args.quanti4 or args.quanti8:
            logger.info("Apply quantization")
            quantization_config = QuantoConfig(
                weights="float8" if args.quantf8 else "int4" if args.quanti4 else "int8",
                # activations="float8" if args.quantf8 else "int8" if args.quanti8 else None
            )
            logger.info("Using 8-bit quantization")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Apply pruning if requested
        if args.prune20 or args.prune40 or args.prune60:
            pruning_amount = 0.2 if args.prune20 else 0.4 if args.prune40 else 0.6
            logger.info(f"Applying unstructured pruning with {pruning_amount*100}% weight removal")
            self.apply_unstructured_pruning(pruning_amount)
        
        logger.info("Model loaded successfully")
    
    def print_model_size(self, model):
        """Calcola la dimensione del modello senza salvare file temporanei"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1e6
        print(f"Size (MB): {size_mb:.2f}")
        return size_mb
    
    def apply_unstructured_pruning(self, pruning_amount: float):
        """
        Apply unstructured magnitude-based pruning to all linear layers
        
        Args:
            pruning_amount: Fraction of weights to prune (0.0 to 1.0)
        """
        logger.info(f"Starting unstructured pruning with {pruning_amount*100}% weight removal")
        
        # Count total linear layers
        linear_layers = []
        total_params_before = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append((name, module))
                total_params_before += module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    total_params_before += module.bias.numel()
        
        logger.info(f"Found {len(linear_layers)} linear layers with {total_params_before:,} total parameters")
        
        # Apply magnitude-based pruning to each linear layer
        pruned_layers = 0
        total_pruned_weights = 0
        
        for name, module in linear_layers:
            try:
                # Prune weights using magnitude-based unstructured pruning
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                total_pruned_weights += int(module.weight.numel() * pruning_amount)
                
                # Also prune bias if it exists
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=pruning_amount)
                    total_pruned_weights += int(module.bias.numel() * pruning_amount)
                
                pruned_layers += 1
                
                if pruned_layers % 50 == 0:  # Log progress every 50 layers
                    logger.info(f"Pruned {pruned_layers}/{len(linear_layers)} linear layers")
                    
            except Exception as e:
                logger.warning(f"Failed to prune layer {name}: {e}")
                continue
        
        logger.info(f"Successfully pruned {pruned_layers}/{len(linear_layers)} linear layers")
        logger.info(f"Total pruned weights: {total_pruned_weights:,}")
        
        # Calculate actual sparsity
        self.calculate_sparsity()
    
    def calculate_sparsity(self):
        """Calculate and log the actual sparsity of the model"""
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check weight sparsity
                if hasattr(module, 'weight_mask'):
                    # Pruned weights (with mask)
                    weight = module.weight_orig * module.weight_mask
                else:
                    # Original weights (not pruned)
                    weight = module.weight
                
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
                
                # Check bias sparsity if exists
                if hasattr(module, 'bias') and module.bias is not None:
                    if hasattr(module, 'bias_mask'):
                        bias = module.bias_orig * module.bias_mask
                    else:
                        bias = module.bias
                    
                    total_params += bias.numel()
                    zero_params += (bias == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Model sparsity: {sparsity*100:.2f}% ({zero_params:,} zero weights out of {total_params:,})")
        return sparsity
    
    def make_pruning_permanent(self):
        """Make the pruning permanent by removing the pruning masks"""
        logger.info("Making pruning permanent by removing masks...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    # Remove weight pruning mask if it exists
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
                    
                    # Remove bias pruning mask if it exists
                    if hasattr(module, 'bias_mask'):
                        prune.remove(module, 'bias')
                        
                except Exception as e:
                    logger.warning(f"Failed to remove pruning mask from {name}: {e}")
        
        logger.info("Pruning masks removed successfully")
    
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
        """
        Create a few-shot prompt for code summarization
        
        Args:
            few_shot_examples: List of validation examples for few-shot learning
            test_code: The code to be summarized
            
        Returns:
            Formatted prompt string
        """
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
        """
        Generate summary using the model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated summary
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4000,  # Leave room for generation
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
    
    def evaluate(self, validation_file: str, test_file: str, num_few_shot: int = 3, num_test_samples: int = 100, output_file: str = None):
        """
        Evaluate the model on test data using few-shot prompting
        
        Args:
            validation_file: Path to validation JSONL file
            test_file: Path to test JSONL file
            num_few_shot: Number of few-shot examples to use
            num_test_samples: Number of test samples to evaluate (None for all)
            output_file: Path to save results
        """
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

        self.print_model_size(self.model)

        ### GPU Warmup
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
            if self.args.quantf8:
                output_file = f"model_llama/code_summarization_results_{self.job_id}_quantf8.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}_quantf8.csv"
            elif self.args.quanti8:
                output_file = f"model_llama/code_summarization_results_{self.job_id}_quanti8.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}_quanti8.csv"
            elif self.args.quanti4:
                output_file = f"model_llama/code_summarization_results_{self.job_id}_quanti4.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}_quanti4.csv"
            elif self.args.prune20:
                output_file = f"model_llama/code_summarization_results_{self.job_id}_prune20.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}_prune20.csv"
            elif self.args.prune40:
                output_file = f"model_llama/code_summarization_results_{self.job_id}_prune40.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}_prune40.csv"
            elif self.args.prune60:
                output_file = f"model_llama/code_summarization_results_{self.job_id}_prune60.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}_prune60.csv"
            else:
                output_file = f"model_llama/code_summarization_results_{self.job_id}.json"
                time_file = f"model_llama/code_summarization_times_{self.job_id}.csv"

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
        logger.info(f"\nEvaluation completed in {total_time/60:.1f} minutes")
        logger.info(f"Results saved to: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Code Summarization with LLaMA-3.1-8B")
    parser.add_argument("--validation_file", 
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl",
                       help="Path to validation JSONL file")
    parser.add_argument("--test_file",
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/test.jsonl", 
                       help="Path to test JSONL file")
    parser.add_argument("--model_name", 
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--num_few_shot", type=int, default=3,
                       help="Number of few-shot examples to use")
    parser.add_argument("--num_test_samples", type=int, default=None,
                       help="Number of test samples to evaluate (None for all)")
    parser.add_argument("--output_file", default=None,
                       help="Output file path for results")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable 8-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature for generation")
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--quantf8", action="store_true")
    parser.add_argument("--quanti8", action="store_true")
    parser.add_argument("--quanti4", action="store_true")
    parser.add_argument("--prune20", action="store_true", 
                       help="Apply unstructured pruning with 20% weight removal")
    parser.add_argument("--prune40", action="store_true",
                       help="Apply unstructured pruning with 40% weight removal") 
    parser.add_argument("--prune60", action="store_true",
                       help="Apply unstructured pruning with 60% weight removal")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Initialize summarizer
    logger.info("Initializing Code Summarizer...")
    summarizer = CodeSummarizer(
        model_name=args.model_name,
        job_id=args.job_id,
        args=args
    )
    
    # Run evaluation
    results = summarizer.evaluate(
        validation_file=args.validation_file,
        test_file=args.test_file,
        num_few_shot=args.num_few_shot,
        num_test_samples=args.num_test_samples,
        output_file=args.output_file
    )
    
    logger.info("Script completed successfully!")
    logger.info(f"Total examples processed: {len(results)}")

if __name__ == "__main__":
    main()
