#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --time=00:05:00


#SBATCH --job-name=test_submission
#SBATCH --output=exercise3pi.out
#SBATCH --error=exercise3pi.err

export OMP_NUM_THREADS=8


### openmp executable
./exercise3pi.exec
