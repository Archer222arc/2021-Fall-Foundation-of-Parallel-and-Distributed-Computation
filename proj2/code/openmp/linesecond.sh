#!/bin/bash
#SBATCH -o linesecond.out
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
gcc -std=c11 linesecond.c -o linesecond -Ofast -march=native -fopenmp -lm -Wall
./linesecond | tee ./results/linesecond.txt

