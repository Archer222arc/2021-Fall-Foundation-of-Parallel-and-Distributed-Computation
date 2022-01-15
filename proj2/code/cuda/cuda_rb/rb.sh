#!/bin/bash
#SBATCH -o rb.out
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
# nvcc -O2 -c rb.cu -o rb -keep -arch sm_20
# ./poisson
# gcc -std=c11 main.c -o rb -Ofast -march=native -fopenmp -lm -Wall -fivopts -ffast-math
# gcc --version
# make
# nvcc -v
nvcc --gpu-architecture=sm_50 -std=c++11 -O3 --use_fast_math -Wno-deprecated-gpu-targets -o rb rb_kernel.cu initialize.cu finalize.cu reduce.cu  test.cu  initial_step.cu residual.cu --compiler-bindir /mnt/lustrefs/softwares/gcc/10.1.0/bin/gcc -w
# nvcc --gpu-architecture=sm_50 -std=c++11 -O3 --use_fast_math -Wno-deprecated-gpu-targets -o rb jacobi_kernel.cu reduce.cu compute_residual.cu test.cu  initial_step.cu 
./rb | tee ./results/rb.txt
rm -dv "./rb" "./rb.o" "./reduce.o" "./initial_step.o" "./compute_residual.o" "./jacobi_kernel.o"
# ./rb

