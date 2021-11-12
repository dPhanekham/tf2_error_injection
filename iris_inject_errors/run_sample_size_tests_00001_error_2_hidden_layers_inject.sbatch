#!/bin/bash
#SBATCH -J sample_size_tests_00001_error_2_hidden_layer
#SBATCH -o batch_output/sample_size_tests_00001_error_2_hidden_layer_%j.out
#SBATCH -D /work/users/dphanekham/tf2_error_injection/iris_inject_errors
#SBATCH -p gpgpu-1
#SBATCH --mail-user dphanekham@smu.edu
#SBATCH --mail-type all

export MODULEPATH=/users/dphanekham/modules:/hpc/modules/applications:/hpc/modules/compilers


module purge
module load modfile

time python sample_size_tests_00001_error_2_hidden_layers_inject.py
