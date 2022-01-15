#!/bin/bash

#SBATCH -o allreduce1.out
#SBATCH --partition=cpu
#SBATCH -J allreduce
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8 
#SBATCH -t 1:59:59

module add mpich
module add gcc
mkdir -p ./results
if [ -f "test" ]; then 
    printf "there is already file compiled\n"
else
    make
fi

for i in {4,5,6,7}
do
mpiexec -n ${i} ./test | tee ./results/process_num'='${i}_slicenum'='commsize*5.txt
done