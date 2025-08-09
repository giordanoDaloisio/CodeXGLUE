#!/usr/bin/env python3
"""
Evaluation script to compute metrics for code summarization results
"""

import json
import argparse
from typing import List, Dict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CodeSummarizationEvaluator:
    def __init__(self):
        """Initialize the evaluator with necessary metrics"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Compute BLEU score between reference and hypothesis
        """
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoothing)
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}")
            return 0.0
    
    def compute_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Compute METEOR score between reference and hypothesis
        """
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            return meteor_score([ref_tokens], hyp_tokens)
        except Exception as e:
            logger.warning(f"Error computing METEOR: {e}")
            return 0.0
    
    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute ROUGE scores between reference and hypothesis
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"Error computing ROUGE: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def evaluate_results(self, results_file: str) -> Dict[str, float]:
        """
        Evaluate the results from the code summarization
        
        Args:
            results_file: Path to JSON file with results
            
        Returns:
            Dictionary with average metrics
        """
        logger.info(f"Loading results from {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Evaluating {len(results)} results...")
        
        bleu_scores = []
        meteor_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        valid_results = 0
        
        for result in results:
            original = result['original_summary'].strip()
            generated = result['generated_summary'].strip()
            
            # Skip results with errors
            if generated.startswith("Error:") or not generated or not original:
                continue
            
            valid_results += 1
            
            # Compute metrics
            bleu = self.compute_bleu(original, generated)
            meteor = self.compute_meteor(original, generated)
            rouge_scores = self.compute_rouge(original, generated)
            
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            rouge1_scores.append(rouge_scores['rouge1'])
            rouge2_scores.append(rouge_scores['rouge2'])
            rougeL_scores.append(rouge_scores['rougeL'])
        
        logger.info(f"Successfully evaluated {valid_results}/{len(results)} results")
        
        # Compute averages
        metrics = {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
            'meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
            'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
            'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0,
            'valid_results': valid_results,
            'total_results': len(results)
        }
        
        return metrics
    
    def print_detailed_analysis(self, results_file: str, num_examples: int = 5):
        """
        Print detailed analysis with examples
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n" + "="*80)
        print("DETAILED ANALYSIS")
        print("="*80)
        
        # Find best and worst examples based on BLEU score
        scored_results = []
        for result in results:
            original = result['original_summary'].strip()
            generated = result['generated_summary'].strip()
            
            if not generated.startswith("Error:") and generated and original:
                bleu = self.compute_bleu(original, generated)
                rouge_scores = self.compute_rouge(original, generated)
                scored_results.append({
                    **result,
                    'bleu': bleu,
                    'rouge1': rouge_scores['rouge1']
                })
        
        if not scored_results:
            logger.error("No valid results found for detailed analysis")
            return
        
        # Sort by BLEU score
        scored_results.sort(key=lambda x: x['bleu'], reverse=True)
        
        print(f"\nBEST {num_examples} RESULTS (by BLEU score):")
        print("-" * 40)
        for i, result in enumerate(scored_results[:num_examples], 1):
            print(f"\n{i}. Function: {result['func_name']}")
            print(f"   BLEU: {result['bleu']:.3f}, ROUGE-1: {result['rouge1']:.3f}")
            print(f"   Original: {result['original_summary']}")
            print(f"   Generated: {result['generated_summary']}")
        
        print(f"\nWORST {num_examples} RESULTS (by BLEU score):")
        print("-" * 40)
        for i, result in enumerate(scored_results[-num_examples:], 1):
            print(f"\n{i}. Function: {result['func_name']}")
            print(f"   BLEU: {result['bleu']:.3f}, ROUGE-1: {result['rouge1']:.3f}")
            print(f"   Original: {result['original_summary']}")
            print(f"   Generated: {result['generated_summary']}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate code summarization results")
    parser.add_argument("results_file", help="Path to JSON file with results")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--num_examples", type=int, default=5, 
                       help="Number of examples to show in detailed analysis")
    
    args = parser.parse_args()
    
    evaluator = CodeSummarizationEvaluator()
    
    # Compute metrics
    metrics = evaluator.evaluate_results(args.results_file)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"BLEU Score:     {metrics['bleu']:.4f}")
    print(f"METEOR Score:   {metrics['meteor']:.4f}")
    print(f"ROUGE-1 Score:  {metrics['rouge1']:.4f}")
    print(f"ROUGE-2 Score:  {metrics['rouge2']:.4f}")
    print(f"ROUGE-L Score:  {metrics['rougeL']:.4f}")
    print(f"\nValid results:  {metrics['valid_results']}/{metrics['total_results']}")
    print("="*50)
    
    # Save metrics
    metrics_file = args.results_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Detailed analysis if requested
    if args.detailed:
        evaluator.print_detailed_analysis(args.results_file, args.num_examples)

if __name__ == "__main__":
    main()
