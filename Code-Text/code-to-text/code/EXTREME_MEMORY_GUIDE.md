# Extreme Memory Optimization Guide for CUDA OOM Issues

## Problem
Even with CPU offloading, you're still getting CUDA out of memory errors during model pruning.

## Root Cause
- PyTorch pruning creates additional mask tensors that consume GPU memory
- Large models like LLaMA-3.1-8B require significant memory even for mask storage
- The original offloading wasn't aggressive enough in memory management

## Solutions (Escalating Order)

### Solution 1: Improved Offloaded Pruning ⚡
**When to use**: If you have 8GB+ GPU memory available
```bash
python3 code_summarization_llama.py --prune20 --offloaded_pruning --num_test_samples 10
```

**Changes made**:
- Reduced group size from 10 to 3 layers
- Added `torch.cuda.synchronize()` for better memory cleanup
- Added memory status logging
- Improved error handling for failed layer transfers

### Solution 2: Combined Gradual + Offloaded Pruning 🔄
**When to use**: If Solution 1 still fails
```bash
python3 code_summarization_llama.py --prune20 --gradual_pruning --offloaded_pruning --num_test_samples 10
```

**How it works**:
- Prunes in 4 smaller steps instead of all at once
- Each step uses aggressive CPU offloading
- Reduces peak memory usage significantly

### Solution 3: CPU-Only Pruning 🛡️ (MOST RELIABLE)
**When to use**: When all other solutions fail
```bash
python3 code_summarization_llama.py --prune20 --cpu_only_pruning --num_test_samples 5
```

**How it works**:
- Moves ENTIRE model to CPU during pruning
- Completely avoids GPU memory issues
- Moves model back to GPU after pruning (if possible)
- ⚠️ **WARNING**: Much slower but guaranteed to work

## Quick Test Commands

### Test 20% Pruning (Safest)
```bash
# Most reliable approach
python3 code_summarization_llama.py --prune20 --cpu_only_pruning --num_test_samples 5

# If you want to try GPU-based approaches first
./test_extreme_memory_optimization.sh
```

### Test 40% or 60% Pruning
```bash
# For higher pruning rates, use CPU-only approach
python3 code_summarization_llama.py --prune40 --cpu_only_pruning --num_test_samples 5
python3 code_summarization_llama.py --prune60 --cpu_only_pruning --num_test_samples 5
```

## Interactive Testing Script
Use the new script that automatically detects your GPU memory and recommends the best strategy:
```bash
./test_extreme_memory_optimization.sh
```

## Memory Monitoring
Monitor GPU memory usage in real-time:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Expected Behavior

### CPU-Only Pruning Timeline:
1. **Model Loading**: Normal GPU loading (~5-10GB)
2. **Pruning Start**: Model moves to CPU, GPU memory drops to ~1-2GB
3. **Pruning Process**: All processing on CPU (slower but safe)
4. **Pruning Complete**: Attempts to move back to GPU
5. **If GPU has space**: Model returns to GPU for inference
6. **If GPU full**: Model stays on CPU for inference

### Performance Impact:
- **GPU Pruning**: Fast (~2-5 minutes)
- **CPU-Only Pruning**: Slower (~10-30 minutes)
- **CPU Inference**: Much slower than GPU (~10x slower)

## Troubleshooting

### If CPU-Only Pruning Also Fails:
This would indicate a system-level issue. Try:
```bash
# Check system memory
free -h

# Ensure no other processes are using GPU
nvidia-smi

# Try with even fewer test samples
python3 code_summarization_llama.py --prune20 --cpu_only_pruning --num_test_samples 2
```

### If Model Won't Fit Back on GPU After Pruning:
The script will automatically keep the model on CPU. You'll see:
```
WARNING: Cannot move pruned model back to GPU - keeping on CPU
WARNING: Model will run on CPU (slower inference)
```

This is normal and the script will continue working, just slower.

## Why This Approach Works

1. **Complete GPU Isolation**: CPU-only pruning completely isolates the pruning process from GPU memory
2. **No Mask Conflicts**: All pruning masks are created and managed in CPU memory
3. **Fallback Safety**: If the pruned model can't fit back on GPU, it gracefully falls back to CPU inference
4. **Memory Monitoring**: Enhanced logging helps identify exactly where memory issues occur

## Recommended Workflow

1. **Start with CPU-only**: `python3 code_summarization_llama.py --prune20 --cpu_only_pruning --num_test_samples 5`
2. **If successful and fast enough**: Continue with CPU-only for full experiments
3. **If too slow**: Try the interactive script to find the fastest approach that works on your system

The CPU-only approach sacrifices speed for reliability - it should work on any system with sufficient RAM (32GB+ recommended for LLaMA-8B).
