#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script per verificare che il setup di Knowledge Distillation funzioni correttamente.
Esegue un mini-test con pochi esempi per validare il codice prima del training completo.

Usage:
    python test_distillation.py
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("TEST KNOWLEDGE DISTILLATION SETUP")
print("=" * 60)

# Test 1: Verifica imports
print("\n[1/5] Test imports...")
try:
    from distillation import DistillationTrainer, CodeSearchNetDataset, load_codesearchnet_python
    print("✓ Imports OK")
except Exception as e:
    print(f"✗ Errore imports: {e}")
    sys.exit(1)

# Test 2: Verifica device
print("\n[2/5] Test device...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test 3: Carica modelli
print("\n[3/5] Test caricamento modelli...")
try:
    print("Caricamento teacher (CodeGPT-small-py)...")
    teacher_tokenizer = AutoTokenizer.from_pretrained('microsoft/CodeGPT-small-py')
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained('microsoft/CodeGPT-small-py')
    print(f"✓ Teacher loaded: {teacher_model.num_parameters():,} params")
    
    print("Caricamento student (DistilGPT-2)...")
    student_model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')
    student_model.resize_token_embeddings(len(teacher_tokenizer))
    print(f"✓ Student loaded: {student_model.num_parameters():,} params")
    
    compression_ratio = student_model.num_parameters() / teacher_model.num_parameters()
    print(f"Compression ratio: {compression_ratio:.2%}")
    
except Exception as e:
    print(f"✗ Errore caricamento modelli: {e}")
    sys.exit(1)

# Test 4: Test tokenization
print("\n[4/5] Test tokenization...")
try:
    test_code = '''"""Calculate factorial of n"""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)'''
    
    tokens = teacher_tokenizer(test_code, return_tensors='pt', truncation=True, max_length=128)
    print(f"✓ Tokenization OK: {tokens['input_ids'].shape[1]} tokens")
    
    # Test forward pass
    with torch.no_grad():
        teacher_output = teacher_model(**tokens)
        student_output = student_model(**tokens)
    
    print(f"✓ Forward pass OK")
    print(f"  Teacher logits shape: {teacher_output.logits.shape}")
    print(f"  Student logits shape: {student_output.logits.shape}")
    
except Exception as e:
    print(f"✗ Errore tokenization/forward: {e}")
    sys.exit(1)

# Test 5: Test DistillationTrainer
print("\n[5/5] Test DistillationTrainer...")
try:
    from transformers import TrainingArguments
    
    # Mini dataset di test
    mini_examples = [
        {
            'func_documentation_string': 'Calculate sum of two numbers',
            'func_code_string': 'def add(a, b):\n    return a + b',
            'func_name': 'add'
        },
        {
            'func_documentation_string': 'Calculate product of two numbers',
            'func_code_string': 'def multiply(a, b):\n    return a * b',
            'func_name': 'multiply'
        }
    ]
    
    mini_dataset = CodeSearchNetDataset(
        mini_examples,
        teacher_tokenizer,
        max_length=128,
        mode='train'
    )
    
    print(f"✓ Dataset creato: {len(mini_dataset)} esempi")
    
    # Test get item
    item = mini_dataset[0]
    print(f"✓ Dataset __getitem__ OK")
    print(f"  Input IDs shape: {len(item['input_ids'])}")
    
    # Test Trainer initialization
    training_args = TrainingArguments(
        output_dir='./test_output',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_steps=1000,
    )
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=2.0,
        alpha=0.5,
        model=student_model,
        args=training_args,
        train_dataset=mini_dataset,
    )
    
    print(f"✓ DistillationTrainer inizializzato")
    print(f"  Temperature: {trainer.temperature}")
    print(f"  Alpha: {trainer.alpha}")
    
except Exception as e:
    print(f"✗ Errore DistillationTrainer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("TUTTI I TEST PASSATI! ✓")
print("=" * 60)
print("\nIl sistema è pronto per la Knowledge Distillation.")
print("\nPer eseguire il training completo:")
print("  bash run_kd_distillation.sh")
print("\nOppure per un test veloce:")
print("  python distillation.py --teacher_model_path microsoft/CodeGPT-small-py \\")
print("                         --student_model_path distilbert/distilgpt2 \\")
print("                         --max_train_samples 1000 \\")
print("                         --max_eval_samples 200 \\")
print("                         --do_train --do_eval")
print("=" * 60)
