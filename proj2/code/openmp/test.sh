#!/bin/bash
#SBATCH -o test.out
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
# gcc -std=c11 jacobi.c -o jacobi -Ofast -march=native -fopenmp -lm -Wall
# ./jacobi
gcc -std=c11 poisson.c -o poisson -Ofast -march=native -fopenmp -lm -Wall
./poisson | tee ./results/baseline.txt

