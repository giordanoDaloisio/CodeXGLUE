#rm logs/*

sbatch test_t5.sh
#sbatch test_distil.sh
sbatch test_prune_t5.sh
sbatch test_prune6_t5.sh
sbatch test_prune4_t5.sh
sbatch test_quant_t5.sh
sbatch test_quantf8_t5.sh
sbatch test_quant4_t5.sh