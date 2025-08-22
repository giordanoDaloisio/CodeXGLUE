#!/usr/bin/env python3
"""
Knowledge Distillation for Code Summarization
Teacher: LLaMA-3.1-8B Instruct
Student: LLaMA-3.2-1B Instruct
"""

import json
import os
import random
import time
import math
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from transformers.optimization import Adafactor
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeSummarizationDataset(Dataset):
    """Dataset for code summarization with teacher-student distillation"""
    
    def __init__(self, data: List[Dict], teacher_tokenizer, student_tokenizer, 
                 few_shot_examples: List[Dict], max_length: int = 1024):
        self.data = data
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.few_shot_examples = few_shot_examples
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def create_few_shot_prompt(self, code: str) -> str:
        """Create few-shot prompt similar to the main script"""
        prompt = """You are an expert software developer tasked with writing concise and accurate summaries for Java methods. Given a Java method, provide a clear, one-sentence summary that describes what the method does.

Here are some examples:

"""
        
        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples, 1):
            example_code = example['code'].strip()
            summary = example['docstring'].strip()
            prompt += f"Example {i}:\nCode:\n```java\n{example_code}\n```\nSummary: {summary}\n\n"
        
        # Add the test case
        prompt += f"Now, please provide a summary for this Java method:\nCode:\n```java\n{code.strip()}\n```\nSummary:"
        
        return prompt
    
    def __getitem__(self, idx):
        item = self.data[idx]
        code = item['code']
        target_summary = item['docstring']
        
        # Create prompt for both teacher and student
        prompt = self.create_few_shot_prompt(code)
        
        # Tokenize for teacher (used to generate soft targets)
        teacher_inputs = self.teacher_tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length - 100,  # Leave room for generation
            padding=False,
            return_tensors="pt"
        )
        
        # Create full sequence for student (prompt + target)
        full_text = prompt + " " + target_summary
        student_inputs = self.student_tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels for student (only supervise the summary part)
        prompt_only = self.student_tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length - 100,  # Ensure consistency
            padding=False,
            return_tensors="pt"
        )
        
        # Labels: -100 for prompt tokens, actual tokens for summary
        labels = student_inputs["input_ids"].clone()
        prompt_length = prompt_only["input_ids"].size(1)
        
        # Ensure we don't go out of bounds
        if prompt_length < labels.size(1):
            labels[0, :prompt_length] = -100
        else:
            # If prompt is longer than full sequence, mask everything
            labels.fill_(-100)
            prompt_length = labels.size(1)
        
        return {
            'teacher_input_ids': teacher_inputs['input_ids'].squeeze(0),
            'teacher_attention_mask': teacher_inputs['attention_mask'].squeeze(0),
            'student_input_ids': student_inputs['input_ids'].squeeze(0),
            'student_attention_mask': student_inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'prompt_length': prompt_length,
            'target_summary': target_summary
        }

class DistillationDataCollator:
    """Custom data collator for distillation"""
    
    def __init__(self, teacher_tokenizer, student_tokenizer):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        
    def __call__(self, batch):
        # Pad teacher inputs
        teacher_input_ids = [item['teacher_input_ids'] for item in batch]
        teacher_attention_mask = [item['teacher_attention_mask'] for item in batch]
        
        teacher_inputs = self.teacher_tokenizer.pad(
            {'input_ids': teacher_input_ids, 'attention_mask': teacher_attention_mask},
            padding=True,
            return_tensors='pt'
        )
        
        # Pad student inputs
        student_input_ids = [item['student_input_ids'] for item in batch]
        student_attention_mask = [item['student_attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        student_inputs = self.student_tokenizer.pad(
            {'input_ids': student_input_ids, 'attention_mask': student_attention_mask},
            padding=True,
            return_tensors='pt'
        )
        
        # Pad labels with -100
        max_length = student_inputs['input_ids'].size(1)
        padded_labels = []
        
        for label in labels:
            if len(label.shape) == 0 or label.size(0) == 0:
                # Handle empty labels
                padded = torch.full((max_length,), -100, dtype=torch.long)
            else:
                padded = torch.full((max_length,), -100, dtype=label.dtype)
                seq_len = min(len(label), max_length)
                padded[:seq_len] = label[:seq_len]
            padded_labels.append(padded)
        
        # Adjust prompt lengths for padding
        adjusted_prompt_lengths = []
        for i, item in enumerate(batch):
            original_length = len(item['student_input_ids'])
            prompt_length = item['prompt_length']
            
            # Ensure prompt length doesn't exceed sequence length
            adjusted_prompt_length = min(prompt_length, original_length, max_length)
            adjusted_prompt_lengths.append(adjusted_prompt_length)
        
        return {
            'teacher_input_ids': teacher_inputs['input_ids'],
            'teacher_attention_mask': teacher_inputs['attention_mask'],
            'student_input_ids': student_inputs['input_ids'],
            'student_attention_mask': student_inputs['attention_mask'],
            'labels': torch.stack(padded_labels),
            'prompt_lengths': adjusted_prompt_lengths,
            'target_summaries': [item['target_summary'] for item in batch]
        }

class KnowledgeDistillationTrainer:
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, teacher_model, student_model, teacher_tokenizer, student_tokenizer,
                 temperature=4.0, alpha=0.7, beta=0.3, device='cuda'):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta    # Weight for hard target loss
        self.device = device
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
    def compute_distillation_loss(self, teacher_logits, student_logits, labels, prompt_lengths):
        """Compute knowledge distillation loss"""
        
        # Soft target loss (KL divergence)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Only compute loss on the summary tokens (not prompt tokens)
        batch_size, seq_len, vocab_size = student_logits.shape
        
        soft_loss = 0
        hard_loss = 0
        valid_tokens = 0
        
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            
            # Extract logits for summary part only
            if prompt_len < seq_len - 1:
                # Get the actual available sequence length for this sample
                available_seq_len = min(seq_len, labels.shape[1])
                
                # Calculate summary region bounds
                summary_start = prompt_len - 1  # Shifted for next token prediction
                summary_end = available_seq_len - 1
                
                if summary_start < summary_end and summary_start >= 0:
                    # Extract logits and labels for summary part
                    summary_teacher_probs = teacher_probs[i, summary_start:summary_end]
                    summary_student_log_probs = student_log_probs[i, summary_start:summary_end]
                    summary_student_logits = student_logits[i, summary_start:summary_end]
                    
                    # Extract corresponding labels (shifted by 1 for next token prediction)
                    label_start = prompt_len
                    label_end = min(label_start + (summary_end - summary_start), labels.shape[1])
                    summary_labels = labels[i, label_start:label_end]
                    
                    # Ensure matching dimensions
                    min_len = min(summary_teacher_probs.shape[0], summary_labels.shape[0])
                    if min_len > 0:
                        summary_teacher_probs = summary_teacher_probs[:min_len]
                        summary_student_log_probs = summary_student_log_probs[:min_len]
                        summary_student_logits = summary_student_logits[:min_len]
                        summary_labels = summary_labels[:min_len]
                        
                        # Soft loss (KL divergence)
                        kl_loss = F.kl_div(summary_student_log_probs, summary_teacher_probs, reduction='none')
                        soft_loss += kl_loss.sum()
                        
                        # Hard loss (cross-entropy with ground truth)
                        valid_mask = (summary_labels != -100)
                        if valid_mask.sum() > 0:
                            hard_loss += F.cross_entropy(
                                summary_student_logits[valid_mask], 
                                summary_labels[valid_mask], 
                                reduction='sum'
                            )
                            valid_tokens += valid_mask.sum()
        
        if valid_tokens > 0:
            soft_loss = soft_loss / valid_tokens
            hard_loss = hard_loss / valid_tokens
        else:
            # Fallback to avoid division by zero
            soft_loss = torch.tensor(0.0, device=teacher_logits.device, requires_grad=True)
            hard_loss = torch.tensor(0.0, device=teacher_logits.device, requires_grad=True)
        
        # Combine losses
        total_loss = self.alpha * (self.temperature ** 2) * soft_loss + self.beta * hard_loss
        
        return total_loss, soft_loss, hard_loss
    
    def train_step(self, batch, optimizer, scheduler=None):
        """Single training step"""
        
        try:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=batch['teacher_input_ids'],
                    attention_mask=batch['teacher_attention_mask']
                )
                teacher_logits = teacher_outputs.logits
            
            # Student forward pass
            student_outputs = self.student_model(
                input_ids=batch['student_input_ids'],
                attention_mask=batch['student_attention_mask']
            )
            student_logits = student_outputs.logits
            
            # Align sequence lengths (teacher might be shorter due to different tokenization)
            min_length = min(teacher_logits.size(1), student_logits.size(1))
            teacher_logits = teacher_logits[:, :min_length, :]
            student_logits = student_logits[:, :min_length, :]
            
            # Adjust prompt lengths to match aligned sequences
            adjusted_prompt_lengths = []
            for prompt_len in batch['prompt_lengths']:
                adjusted_prompt_lengths.append(min(prompt_len, min_length))
            
            # Compute distillation loss
            total_loss, soft_loss, hard_loss = self.compute_distillation_loss(
                teacher_logits, student_logits, batch['labels'], adjusted_prompt_lengths
            )
            
            # Check for invalid loss values
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("Invalid loss detected, skipping this batch")
                return {
                    'total_loss': 0.0,
                    'soft_loss': 0.0,
                    'hard_loss': 0.0
                }
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            return {
                'total_loss': total_loss.item(),
                'soft_loss': soft_loss.item(),
                'hard_loss': hard_loss.item()
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            # Return zero losses to continue training
            return {
                'total_loss': 0.0,
                'soft_loss': 0.0,
                'hard_loss': 0.0
            }

class CodeSummarizationDistiller:
    """Main class for knowledge distillation in code summarization"""
    
    def __init__(self, 
                 teacher_model_name="meta-llama/Llama-3.1-8B-Instruct",
                 student_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 temperature=4.0,
                 alpha=0.7,
                 beta=0.3):
        
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load teacher model and tokenizer
        logger.info(f"Loading teacher model: {teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Load student model and tokenizer
        logger.info(f"Loading student model: {student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
            
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Initialize distillation trainer
        self.distiller = KnowledgeDistillationTrainer(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            teacher_tokenizer=self.teacher_tokenizer,
            student_tokenizer=self.student_tokenizer,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            device=self.device
        )
        
        logger.info("Models loaded successfully")
    
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
    
    def prepare_datasets(self, train_file: str, validation_file: str, max_train_samples=None):
        """Prepare training and validation datasets"""
        
        # Load data
        train_data = self.load_jsonl(train_file)
        validation_data = self.load_jsonl(validation_file)
        
        # Use validation data for few-shot examples
        few_shot_examples = random.sample(validation_data, min(3, len(validation_data)))
        
        # Limit training samples if specified
        if max_train_samples and max_train_samples < len(train_data):
            train_data = random.sample(train_data, max_train_samples)
            logger.info(f"Using {max_train_samples} training samples")
        
        # Create datasets
        train_dataset = CodeSummarizationDataset(
            data=train_data,
            teacher_tokenizer=self.teacher_tokenizer,
            student_tokenizer=self.student_tokenizer,
            few_shot_examples=few_shot_examples
        )
        
        # Use a small portion of train data for validation during training
        val_data = random.sample(train_data, min(100, len(train_data) // 10))
        val_dataset = CodeSummarizationDataset(
            data=val_data,
            teacher_tokenizer=self.teacher_tokenizer,
            student_tokenizer=self.student_tokenizer,
            few_shot_examples=few_shot_examples
        )
        
        return train_dataset, val_dataset, few_shot_examples
    
    def train(self, train_file: str, validation_file: str, 
              output_dir: str = "./distilled_model",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5,
              max_train_samples: int = None,
              save_steps: int = 500,
              eval_steps: int = 500,
              warmup_ratio: float = 0.1):
        """Train the student model using knowledge distillation"""
        
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, few_shot_examples = self.prepare_datasets(
            train_file, validation_file, max_train_samples
        )
        
        # Create data collator
        data_collator = DistillationDataCollator(
            teacher_tokenizer=self.teacher_tokenizer,
            student_tokenizer=self.student_tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=data_collator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator
        )
        
        # Setup optimizer and scheduler
        optimizer = Adafactor(self.student_model.parameters(), 
            scale_parameter=False, 
            relative_step=False, 
            warmup_init=False, 
            lr=learning_rate)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Total training steps: {total_steps}")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.student_model.train()
            epoch_losses = {'total': [], 'soft': [], 'hard': []}
            
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch in train_pbar:
                losses = self.distiller.train_step(batch, optimizer, scheduler)
                
                epoch_losses['total'].append(losses['total_loss'])
                epoch_losses['soft'].append(losses['soft_loss'])
                epoch_losses['hard'].append(losses['hard_loss'])
                
                global_step += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'total_loss': f"{losses['total_loss']:.4f}",
                    'soft_loss': f"{losses['soft_loss']:.4f}",
                    'hard_loss': f"{losses['hard_loss']:.4f}"
                })
                
                # Validation and saving
                if global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_loader)
                    logger.info(f"Step {global_step} - Validation loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(output_dir, f"best_model_step_{global_step}")
                        logger.info(f"New best model saved at step {global_step}")
                
                if global_step % save_steps == 0:
                    self.save_model(output_dir, f"checkpoint_step_{global_step}")
            
            # Log epoch statistics
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            logger.info(f"Epoch {epoch + 1} - Avg Total Loss: {avg_losses['total']:.4f}, "
                       f"Avg Soft Loss: {avg_losses['soft']:.4f}, "
                       f"Avg Hard Loss: {avg_losses['hard']:.4f}")
        
        # Save final model
        self.save_model(output_dir, "final_model")
        logger.info("Training completed!")
        
        return output_dir
    
    def evaluate(self, val_loader):
        """Evaluate the model on validation set"""
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Teacher forward pass
                teacher_outputs = self.teacher_model(
                    input_ids=batch['teacher_input_ids'],
                    attention_mask=batch['teacher_attention_mask']
                )
                teacher_logits = teacher_outputs.logits
                
                # Student forward pass
                student_outputs = self.student_model(
                    input_ids=batch['student_input_ids'],
                    attention_mask=batch['student_attention_mask']
                )
                student_logits = student_outputs.logits
                
                # Align sequence lengths
                min_length = min(teacher_logits.size(1), student_logits.size(1))
                teacher_logits = teacher_logits[:, :min_length, :]
                student_logits = student_logits[:, :min_length, :]
                
                # Compute loss
                loss, _, _ = self.distiller.compute_distillation_loss(
                    teacher_logits, student_logits, batch['labels'], batch['prompt_lengths']
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.student_model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_model(self, output_dir: str, checkpoint_name: str):
        """Save the student model and tokenizer"""
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.student_model.save_pretrained(checkpoint_dir)
        self.student_tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training configuration
        config = {
            'teacher_model': self.teacher_model_name,
            'student_model': self.student_model_name,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        with open(os.path.join(checkpoint_dir, 'distillation_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Code Summarization")
    
    # Data arguments
    parser.add_argument("--train_file", 
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/train.jsonl",
                       help="Path to training JSONL file")
    parser.add_argument("--validation_file",
                       default="/NFSHOME/gdaloisio/code/CodeXGLUE/Code-Text/code-to-text/dataset/java/valid.jsonl",
                       help="Path to validation JSONL file")
    
    # Model arguments
    parser.add_argument("--teacher_model",
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Teacher model name")
    parser.add_argument("--student_model",
                       default="meta-llama/Llama-3.2-1B-Instruct", 
                       help="Student model name")
    
    # Training arguments
    parser.add_argument("--output_dir",
                       default="./distilled_llama_code_summarization",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples to use")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Temperature for knowledge distillation")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Weight for soft targets (distillation loss)")
    parser.add_argument("--beta", type=float, default=0.3,
                       help="Weight for hard targets (ground truth loss)")
    
    # Training control
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio for learning rate scheduler")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize distiller
    logger.info("Initializing Knowledge Distillation for Code Summarization...")
    distiller = CodeSummarizationDistiller(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # Start training
    output_dir = distiller.train(
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio
    )
    
    logger.info(f"Knowledge distillation completed! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
