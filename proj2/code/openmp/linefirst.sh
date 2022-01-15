#!/bin/bash
#SBATCH -o linefirst.out
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
gcc -std=c11 linefirst.c -o linefirst -Ofast -march=native -fopenmp -lm -Wall
./linefirst | tee ./results/linefirst.txt

