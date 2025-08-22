# Knowledge Distillation for Code Summarization

This directory contains scripts for performing knowledge distillation from **LLaMA-3.1-8B Instruct** (teacher) to **LLaMA-3.2-1B Instruct** (student) for the code summarization task.

## Overview

Knowledge distillation is a technique where a smaller "student" model learns to mimic the behavior of a larger "teacher" model. This allows us to create more efficient models that retain much of the performance of the larger model while being faster and requiring less memory.

### Models
- **Teacher**: LLaMA-3.1-8B Instruct (~8 billion parameters, ~16GB)
- **Student**: LLaMA-3.2-1B Instruct (~1 billion parameters, ~2GB)
- **Expected Compression**: ~8x smaller model, ~5-10x faster inference

## Files

### Main Scripts
- `knowledge_distillation_llama.py` - Main training script for knowledge distillation
- `evaluate_distilled_model.py` - Evaluation script for the distilled model
- `compare_teacher_student.py` - Compare performance between teacher and student models
- `train_distillation.sh` - Interactive training script with different configurations

### Generated Outputs
- `./distilled_model_*/` - Trained model checkpoints and final models
- `comparison_results/` - Performance comparison between teacher and student

## Quick Start

### 1. Train a Distilled Model (Quick Test)
```bash
# Interactive training script
./train_distillation.sh

# Or run directly with quick test settings
python3 knowledge_distillation_llama.py \
    --max_train_samples 100 \
    --num_epochs 1 \
    --batch_size 2 \
    --output_dir "./distilled_model_quick_test"
```

### 2. Evaluate the Distilled Model
```bash
# Evaluate on test set
python3 evaluate_distilled_model.py \
    --model_path "./distilled_model_quick_test/best_model_step_50" \
    --num_test_samples 50
```

### 3. Compare Teacher vs Student Performance
```bash
# Compare both models on the same test set
python3 compare_teacher_student.py \
    --student_model_path "./distilled_model_quick_test/best_model_step_50" \
    --num_test_samples 50
```

## Training Configurations

### Quick Test (30 minutes)
- 100 training samples
- 1 epoch
- Batch size: 2
- Good for testing the pipeline

### Small Training (2-3 hours)
- 1,000 training samples
- 2 epochs
- Batch size: 4
- Good for initial experiments

### Medium Training (8-10 hours)
- 5,000 training samples
- 3 epochs
- Batch size: 4
- Good for production models

### Full Training (1-2 days)
- All training samples (~160k)
- 3-5 epochs
- Batch size: 4-8
- Best performance

## Command Line Options

### Training (`knowledge_distillation_llama.py`)
```bash
python3 knowledge_distillation_llama.py \
    --train_file /path/to/train.jsonl \
    --validation_file /path/to/valid.jsonl \
    --teacher_model meta-llama/Llama-3.1-8B-Instruct \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    --output_dir ./distilled_model \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --max_train_samples 1000 \
    --temperature 4.0 \
    --alpha 0.7 \
    --beta 0.3
```

**Key Parameters:**
- `--temperature`: Controls the softness of teacher predictions (higher = softer)
- `--alpha`: Weight for distillation loss (soft targets)
- `--beta`: Weight for supervised loss (hard targets)
- `--max_train_samples`: Limit training data for faster experimentation

### Evaluation (`evaluate_distilled_model.py`)
```bash
python3 evaluate_distilled_model.py \
    --model_path ./distilled_model/best_model_step_500 \
    --num_test_samples 100 \
    --output_file distilled_results.json
```

### Comparison (`compare_teacher_student.py`)
```bash
python3 compare_teacher_student.py \
    --student_model_path ./distilled_model/best_model_step_500 \
    --num_test_samples 50 \
    --output_dir ./comparison_results
```

## Expected Results

Based on typical knowledge distillation results:

### Performance Retention
- **BLEU Score**: 80-90% of teacher performance
- **ROUGE Score**: 85-95% of teacher performance
- **Human Evaluation**: Similar quality summaries

### Efficiency Gains
- **Model Size**: ~8x smaller (16GB → 2GB)
- **Inference Speed**: ~5-10x faster
- **Memory Usage**: ~8x less GPU memory required

### Example Comparison
| Model | Size | Parameters | Avg Time/Sample | BLEU-4 |
|-------|------|------------|-----------------|---------|
| Teacher (LLaMA-3.1-8B) | 16GB | 8B | 2.5s | 0.35 |
| Student (LLaMA-3.2-1B) | 2GB | 1B | 0.4s | 0.31 |
| **Improvement** | **8x smaller** | **8x fewer** | **6x faster** | **88% retained** |

## Troubleshooting

### CUDA Out of Memory
1. Reduce batch size: `--batch_size 1` or `--batch_size 2`
2. Use gradient accumulation (modify the code)
3. Use CPU offloading for teacher model

### Slow Training
1. Reduce training samples: `--max_train_samples 500`
2. Reduce sequence length in the dataset class
3. Use mixed precision training (modify the code)

### Poor Student Performance
1. Increase training data: remove `--max_train_samples`
2. Increase epochs: `--num_epochs 5`
3. Adjust distillation parameters:
   - Lower temperature: `--temperature 2.0`
   - Balance loss weights: `--alpha 0.5 --beta 0.5`

### Model Loading Issues
1. Check internet connection for model downloads
2. Ensure sufficient disk space for model cache
3. Verify Hugging Face authentication if using gated models

## Monitoring Training

### View Training Progress
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f training_log.txt
```

### Check Model Outputs
Training saves checkpoints at regular intervals:
- `best_model_step_*` - Best performing checkpoint
- `checkpoint_step_*` - Regular training checkpoints
- `final_model` - Final model after all epochs

## Advanced Usage

### Custom Loss Functions
Modify `compute_distillation_loss` in `knowledge_distillation_llama.py` to experiment with:
- Different distance metrics (MSE, cosine similarity)
- Feature-level distillation
- Attention transfer

### Hyperparameter Tuning
Key hyperparameters to tune:
- `temperature` (2.0 - 8.0): Higher for more diverse learning
- `alpha/beta` (0.3-0.9): Balance between soft and hard targets
- `learning_rate` (1e-5 - 1e-4): Adjust based on convergence

### Multi-GPU Training
For larger datasets, modify the code to use:
- `torch.nn.DataParallel`
- `torch.distributed`
- Hugging Face `Accelerate`

## Citations

If you use this knowledge distillation implementation, please cite:

```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```
