#rm logs/*

sbatch test.sh
sbatch test_distil.sh
sbatch test_prune.sh
sbatch test_prune4.sh
sbatch test_prune6.sh
sbatch test_quant.sh
sbatch test_quant4.sh
sbatch test_quantf8.sh