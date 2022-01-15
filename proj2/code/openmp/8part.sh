#!/bin/bash
#SBATCH -o 8part.out
#SBATCH --partition=cpu
#SBATCH -J allreduce
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 1:59:59

module add mpich
module add gcc
mkdir -p ./results
# if [ -f "poisson" ]; then 
#     printf "there is already file compiled\n"
# else
#     make
# fi
# make
# ./poisson
gcc -std=c11 8part.c -o 8part -Ofast -march=native -fopenmp -lm -Wall -fivopts -ffast-math
./8part | tee ./results/8part.txt

