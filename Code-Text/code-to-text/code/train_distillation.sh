#!/bin/bash

# Quick Knowledge Distillation Test Script
# Train LLaMA-3.2-1B using knowledge from LLaMA-3.1-8B for code summarization

echo "🎓 Knowledge Distillation: LLaMA-3.1-8B → LLaMA-3.2-1B"
echo "======================================================="

# Check if we're in the right directory
if [ ! -f "knowledge_distillation_llama.py" ]; then
    echo "❌ Error: knowledge_distillation_llama.py not found in current directory"
    echo "Please run this script from the code directory"
    exit 1
fi

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Current GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print "   Used: " $1 "MB / Total: " $2 "MB"}'
    
    AVAILABLE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    echo "   Available: ${AVAILABLE_MEMORY}MB"
    echo ""
    
    if [ "$AVAILABLE_MEMORY" -lt 12000 ]; then
        echo "⚠️  WARNING: Less than 12GB GPU memory available."
        echo "   Knowledge distillation requires substantial memory for both teacher and student models."
        echo "   Consider reducing batch size or using CPU offloading."
        echo ""
    fi
else
    echo "⚠️  nvidia-smi not found. Cannot check GPU memory."
    echo ""
fi

echo "🎯 Training Options:"
echo "==================="
echo "1. Quick Test (100 samples, 1 epoch) - ~30 minutes"
echo "2. Small Training (1000 samples, 2 epochs) - ~2-3 hours" 
echo "3. Medium Training (5000 samples, 3 epochs) - ~8-10 hours"
echo "4. Custom Configuration"
echo "5. Exit"
echo ""

read -p "Select option [1-5]: " choice

case $choice in
    1)
        echo "🚀 Starting Quick Test Training..."
        python3 knowledge_distillation_llama.py \
            --max_train_samples 100 \
            --num_epochs 1 \
            --batch_size 2 \
            --eval_steps 50 \
            --save_steps 50 \
            --output_dir "./distilled_model_quick_test" \
            --learning_rate 5e-5 \
            --temperature 4.0 \
            --alpha 0.7 \
            --beta 0.3
        ;;
    2)
        echo "🚀 Starting Small Training..."
        python3 knowledge_distillation_llama.py \
            --max_train_samples 1000 \
            --num_epochs 2 \
            --batch_size 4 \
            --eval_steps 100 \
            --save_steps 200 \
            --output_dir "./distilled_model_small" \
            --learning_rate 5e-5 \
            --temperature 4.0 \
            --alpha 0.7 \
            --beta 0.3
        ;;
    3)
        echo "🚀 Starting Medium Training..."
        python3 knowledge_distillation_llama.py \
            --max_train_samples 5000 \
            --num_epochs 3 \
            --batch_size 4 \
            --eval_steps 250 \
            --save_steps 500 \
            --output_dir "./distilled_model_medium" \
            --learning_rate 3e-5 \
            --temperature 4.0 \
            --alpha 0.7 \
            --beta 0.3
        ;;
    4)
        echo "📝 Custom Configuration:"
        echo ""
        read -p "Number of training samples (default: 1000): " samples
        samples=${samples:-1000}
        
        read -p "Number of epochs (default: 2): " epochs
        epochs=${epochs:-2}
        
        read -p "Batch size (default: 4): " batch_size
        batch_size=${batch_size:-4}
        
        read -p "Learning rate (default: 5e-5): " lr
        lr=${lr:-5e-5}
        
        read -p "Temperature (default: 4.0): " temp
        temp=${temp:-4.0}
        
        read -p "Output directory (default: ./distilled_model_custom): " output_dir
        output_dir=${output_dir:-"./distilled_model_custom"}
        
        echo "🚀 Starting Custom Training..."
        python3 knowledge_distillation_llama.py \
            --max_train_samples $samples \
            --num_epochs $epochs \
            --batch_size $batch_size \
            --eval_steps 100 \
            --save_steps 200 \
            --output_dir "$output_dir" \
            --learning_rate $lr \
            --temperature $temp \
            --alpha 0.7 \
            --beta 0.3
        ;;
    5)
        echo "👋 Exiting..."
        exit 0
        ;;
    *)
        echo "❌ Invalid option. Please select 1-5."
        exit 1
        ;;
esac

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Knowledge distillation completed successfully!"
    echo ""
    echo "📁 Model saved to: $output_dir"
    echo ""
    echo "🔍 To evaluate the distilled model, run:"
    echo "python3 evaluate_distilled_model.py --model_path $output_dir/best_model_step_* --num_test_samples 50"
    echo ""
    echo "📊 Training logs and checkpoints are available in the output directory."
    echo ""
    echo "💡 Next steps:"
    echo "1. Evaluate the distilled model performance"
    echo "2. Compare with the original teacher model"
    echo "3. Fine-tune hyperparameters if needed"
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    echo ""
    echo "🔧 Common issues:"
    echo "- CUDA out of memory: Try reducing batch_size to 2 or 1"
    echo "- Model loading errors: Check internet connection for model downloads"
    echo "- Dataset errors: Verify the dataset files exist and are readable"
fi
