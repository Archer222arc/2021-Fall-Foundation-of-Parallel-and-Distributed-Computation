# CC = gcc
CFLAGS = -std=c99
export CC = /mnt/lustrefs/softwares/gcc/9.4.0
NVCC = nvcc
# NVCC_FLAGS = --gpu-architecture=sm_50 -std=c++11 -O3 -Wno-deprecated-gpu-targets -allow-unsupported-compiler 
NVCC_FLAGS = --gpu-architecture=sm_50 -std=c++11 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm

jacobi: main.o jacobi_kernel.o initial_step.o compute_residual.o reduce.o
	$(CC) $^ -o $@ $(LIBRARIES)

# main.o: main.c
# 	$(CC) $(CFLAGS) -c $^ -o $@

compute_residual.o: compute_residual.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

reduce.o: reduce.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

jacobi_kernel.o: jacobi_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
	
initial_step.o: initial_step.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
main.o: test.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
clean:
	rm -f *.o jacobi
# CC = gcc
# CFLAGS = -std=c99
# $(CC) $(CFLAGS) -c $^ -o $@

# NVCC = nvcc
# NVCC_FLAGS = --gpu-architecture=sm_50 -std=c++11 -O3 -Wno-deprecated-gpu-targets

# LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm
# all: jacobi
# jacobi: jacobi.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
	