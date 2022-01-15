#include "head.h"
#include "cuda.h"

__device__ void inReduce3(volatile double *uds, int index);

__global__
void reduce1(const double* res, double *res0){
  // dimz > dimy > dimx 
  int tx = threadIdx.x;
  __shared__ double uds[rsize];
  uds[tx] = res[blockIdx.x * blockDim.x + tx];
  __syncthreads();
  for (int idx = blockDim.x / 2; idx >32 ; idx >>= 1){
    if (tx < idx) 
      uds[tx] += uds[idx + tx];
    __syncthreads();
  }
  // __syncthreads();
  if (tx < 32){
    inReduce3(uds, tx);
  }
  if (tx == 0)
    res0[blockIdx.x] = uds[0];
}

__global__
void reduce2(const double* res, double *res0){
  // dimz > dimy > dimx 
  int tx = threadIdx.x;
  __shared__ double uds[rstep];
  uds[tx] = res[tx];
  __syncthreads();
  for (int idx = blockDim.x / 2; idx >32 ; idx >>= 1){
    if (tx < idx) 
      uds[tx] += uds[idx + tx];
    __syncthreads();
  }
  __syncthreads();
  if (tx < 32){
    inReduce3(uds, tx);
  }
  if (tx == 0)
    res0[blockIdx.x] = uds[0];
}

__device__ void inReduce3(volatile double *uds, int index){
  uds[index] += uds[index + 32];
  uds[index] += uds[index + 16];
  uds[index] += uds[index + 8];
  uds[index] += uds[index + 4];
  uds[index] += uds[index + 2];
  uds[index] += uds[index + 1];
}

