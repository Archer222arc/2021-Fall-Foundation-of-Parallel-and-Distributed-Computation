#!/bin/bash

#SBATCH -o allreduce_circle.out
#SBATCH --partition=cpu
#SBATCH -J allreduce
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8 
#SBATCH -t 1:59:59

module add mpich
module add gcc
mkdir -p ./results

for i in {4,5,6,7}
do
mpicc allreduce_circle.c -o allreduce_circle -lm
mpiexec -n ${i} ./allreduce_circle | tee ./results/circle_process_num'='${i}_slicenum'='commsize.txt
done