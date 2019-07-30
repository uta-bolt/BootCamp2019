#!/bin/bash -l

#SBATCH --ntasks=16

#SBATCH --time=00:02:00

#SBATCH --output=mpi_bc.out
#SBATCH --error=mpi_bc.err


### MPI executable
mpiexec -np $SLURM_NTASKS ./broadcast.exec
