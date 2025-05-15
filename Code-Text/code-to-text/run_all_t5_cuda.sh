#rm logs/*

sbatch test_cuda_t5.sh
#sbatch test_cuda_distil.sh
sbatch test_cuda_prune_t5.sh
sbatch test_cuda_prune6_t5.sh
sbatch test_cuda_prune4_t5.sh
sbatch test_cuda_quant_t5.sh
sbatch test_cuda_quant8_t5.sh
sbatch test_cuda_quant4_t5.sh