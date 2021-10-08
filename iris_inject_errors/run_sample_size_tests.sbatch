#!/bin/bash
#SBATCH -J batch_test
#SBATCH -o python_%j.out
#SBATCH -D /work/users/dphanekham
#SBATCH -p gpgpu-1

export MODULEPATH=/users/dphanekham/modules:/hpc/modules/applications:/hpc/modules/compilers


module purge
module load modfile

time python batch_test.py
