rm logs/*

sbatch test.sh
sbatch test_distil.sh
sbatch test_quant.sh
sbatch test_prune.sh
sbatch test_cuda.sh
sbatch test_cuda_quant.sh
sbatch test_cuda_prune.sh
sbatch test_cuda_distil.sh