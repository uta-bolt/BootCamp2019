#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --time=00:05:00


#SBATCH --job-name=test_submission
#SBATCH --output=openmp_comparison.out
#SBATCH --error=openmp_comparison.err

export OMP_NUM_THREADS=1
### openmp executable
./dot_prod_ex2.exec
./dot_prod_ex2_v3.exec
#SBATCH --output=openmp_test_thread4.out
#SBATCH --error=openmp_test_v3.err

export OMP_NUM_THREADS=4
### openmp executable
./dot_prod_ex2.exec
./dot_prod_ex2_v3.exec
#SBATCH --output=openmp_test_thread8.out
#SBATCH --error=openmp_test_v8.err

export OMP_NUM_THREADS=8
### openmp executable
./dot_prod_ex2.exec
./dot_prod_ex2_v3.exec
