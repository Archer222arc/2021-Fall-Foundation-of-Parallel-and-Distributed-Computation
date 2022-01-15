#!/bin/bash
#SBATCH -o jacobi.out
#SBATCH --partition=gpu
#SBATCH -J allreduce
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH -t 1:59:59

module add mpich
module add gcc
module add cuda/11.1
mkdir -p ./results
# if [ -f "poisson" ]; then 
#     printf "there is already file compiled\n"
# else
#     make
# fi
# nvcc -O2 -c jacobi.cu -o jacobi -keep -arch sm_20
# ./poisson
# gcc -std=c11 main.c -o jacobi -Ofast -march=native -fopenmp -lm -Wall -fivopts -ffast-math
# gcc --version
# make
# nvcc -v
nvcc --gpu-architecture=sm_50 -std=c++11 -O3 --use_fast_math -Wno-deprecated-gpu-targets -o jacobi jacobi_kernel.cu reduce.cu  test.cu  initial_step.cu residual.cu --compiler-bindir /mnt/lustrefs/softwares/gcc/10.1.0/bin/gcc -w
# nvcc --gpu-architecture=sm_50 -std=c++11 -O3 --use_fast_math -Wno-deprecated-gpu-targets -o jacobi jacobi_kernel.cu reduce.cu compute_residual.cu test.cu  initial_step.cu 
./jacobi | tee ./results/jacobi.txt
rm -dv "./jacobi" "./jacobi.o" "./reduce.o" "./initial_step.o" "./compute_residual.o" "./jacobi_kernel.o"
# ./jacobi

