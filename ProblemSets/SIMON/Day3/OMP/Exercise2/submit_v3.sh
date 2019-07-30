#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --time=00:01:00


#SBATCH --job-name=test_submission
#SBATCH --output=openmp_test_v3.out
#SBATCH --error=openmp_test_v3.err

export OMP_NUM_THREADS=8


### openmp executable
./dot_prod_ex2_v3.exec
