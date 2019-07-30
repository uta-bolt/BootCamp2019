#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=00:01:00


#SBATCH --job-name=test_submission
#SBATCH --output=openmp_test1_v1.out
#SBATCH --error=openmp_test.err

export OMP_NUM_THREADS=4


### openmp executable
./dot_prod_ex2.exec
