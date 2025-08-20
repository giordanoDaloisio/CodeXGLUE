#!/usr/bin/env python3
"""
Simple test script to verify pruning functionality
"""

import torch
import torch.nn.utils.prune as prune
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pruning_functionality():
    """Test the pruning functionality with a small model"""
    
    logger.info("Testing pruning functionality...")
    
    # Use a smaller model for testing
    model_name = "microsoft/DialoGPT-small"  # Much smaller than LLaMA for testing
    
    try:
        logger.info(f"Loading test model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Model loaded successfully")
        
        # Count linear layers before pruning
        linear_layers = []
        total_params_before = 0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append((name, module))
                total_params_before += module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    total_params_before += module.bias.numel()
        
        logger.info(f"Found {len(linear_layers)} linear layers with {total_params_before:,} parameters")
        
        # Apply 20% pruning
        pruning_amount = 0.2
        logger.info(f"Applying {pruning_amount*100}% unstructured pruning...")
        
        pruned_layers = 0
        for name, module in linear_layers:
            try:
                # Prune weights
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                
                # Prune bias if exists
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=pruning_amount)
                
                pruned_layers += 1
                
            except Exception as e:
                logger.warning(f"Failed to prune layer {name}: {e}")
                continue
        
        logger.info(f"Successfully pruned {pruned_layers}/{len(linear_layers)} layers")
        
        # Calculate sparsity
        total_params = 0
        zero_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check weight sparsity
                if hasattr(module, 'weight_mask'):
                    weight = module.weight_orig * module.weight_mask
                else:
                    weight = module.weight
                
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
                
                # Check bias sparsity
                if hasattr(module, 'bias') and module.bias is not None:
                    if hasattr(module, 'bias_mask'):
                        bias = module.bias_orig * module.bias_mask
                    else:
                        bias = module.bias
                    
                    total_params += bias.numel()
                    zero_params += (bias == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Achieved sparsity: {sparsity*100:.2f}% ({zero_params:,} zero weights out of {total_params:,})")
        
        # Test inference
        logger.info("Testing inference with pruned model...")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test inference successful. Input: '{test_input}' -> Output: '{response}'")
        
        # Make pruning permanent
        logger.info("Making pruning permanent...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
                    if hasattr(module, 'bias_mask'):
                        prune.remove(module, 'bias')
                except Exception as e:
                    logger.warning(f"Failed to remove mask from {name}: {e}")
        
        logger.info("Pruning test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pruning test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pruning_functionality()
    if success:
        print("\n✅ Pruning functionality test PASSED")
    else:
        print("\n❌ Pruning functionality test FAILED")
