#include "head.h"

__device__ void inReduce2(volatile double *uds, int index);

__global__
void residual(const double* u, const double *u1, const double* b, double *res, int count){

  __shared__ double uds[blksize];
  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Hei = bidz * blockDim.z + tidz;
  int Col = bidy * blockDim.y + tidy;
  int Row = bidx * blockDim.x + tidx;

  // int Row1 = (bidx * blockDim.x + tidx) * 2 + (Hei+Col+count+1)%2;

  int index = tidx + tidy * blockDim.x + tidz * blockDim.x * blockDim.y;
  double tmp;
  int cnt = (Col+Hei+count)%2;
  tmp = b[Hei * NN/2 + Col * N/2 + Row]
    + u1[(Hei+1) * N22N + (Col + 1)*N22 + Row+cnt]
    + u1[(Hei+1) * N22N + (Col + 1)*N22 + Row+cnt+1]
    + u1[(Hei+1) * N22N + Col * N22 + Row + 1]
    + u1[(Hei+1) * N22N + (Col + 2) * N22 + Row + 1]
    + u1[Hei * N22N + (Col + 1) * N22 + Row + 1]
    + u1[(Hei+2) * N22N + (Col + 1) * N22 + Row + 1];

  double res0 =  tmp - 6 * u[(Hei+1) * N22N + (Col + 1) * N22 + Row + 1];

  uds[index] = res0 * res0;
  __syncthreads();

  for (int idx = blksize/2; idx > 32; idx >>= 1){

    if (index < idx)
      uds[index] += uds[index + idx];
    __syncthreads();

  }

  if (index < 32){
    inReduce2(uds, index);
  }

  if (index == 0)
    res[bidx + bidy * nx + bidz * nx * ny] = uds[0];
  // if (index == 0)
    // res[(2*bidx + count) + bidy * nx + bidz * nx * ny] = uds[0];
}


__device__ void inReduce2(volatile double *uds, int index){
    uds[index] += uds[index + 32];
    uds[index] += uds[index + 16];
    uds[index] += uds[index + 8];
    uds[index] += uds[index + 4];
    uds[index] += uds[index + 2];
    uds[index] += uds[index + 1];
}

