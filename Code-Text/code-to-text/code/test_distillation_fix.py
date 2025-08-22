#!/usr/bin/env python3
"""
Test script for knowledge distillation with bug fixes
"""

import torch
import logging
from knowledge_distillation_llama import CodeSummarizationDistiller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_distillation():
    """Test the distillation pipeline with minimal data"""
    
    logger.info("🧪 Testing Knowledge Distillation Pipeline...")
    
    try:
        # Initialize distiller with small models for testing
        distiller = CodeSummarizationDistiller(
            teacher_model_name="meta-llama/Llama-3.1-8B-Instruct",
            student_model_name="meta-llama/Llama-3.2-1B-Instruct",
            temperature=4.0,
            alpha=0.7,
            beta=0.3
        )
        
        logger.info("✅ Models loaded successfully")
        
        # Test with very small dataset
        logger.info("🏃 Starting minimal training test...")
        
        output_dir = distiller.train(
            train_file="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/train.jsonl",
            validation_file="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl",
            output_dir="./test_distillation_fix",
            num_epochs=1,
            batch_size=1,  # Very small batch size
            learning_rate=5e-5,
            max_train_samples=5,  # Only 5 samples
            save_steps=2,
            eval_steps=2,
            warmup_ratio=0.1
        )
        
        logger.info(f"✅ Test training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distillation()
    if success:
        print("\n🎉 Knowledge distillation fix verified!")
        print("You can now run the full training with:")
        print("./train_distillation.sh")
    else:
        print("\n💥 Fix verification failed. Check the error messages above.")
