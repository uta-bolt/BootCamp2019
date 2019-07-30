#!/bin/bash -l

#SBATCH --ntasks=10

#SBATCH --time=00:02:00

#SBATCH --output=mc_mpi.out
#SBATCH --error=mc_mpi.err


### MPI executable
mpiexec -np $SLURM_NTASKS ./mc_mpi.exec
