#include "head.h"

__device__ void inReduce1(volatile double *uds, int index);

// simple read/load
__global__
void rb_kernel(double* u, const double* b, double *res, int count){

  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Hei = bidz * blockDim.z + tidz;
  int Col = bidy * blockDim.y + tidy;
  int Row = (bidx * blockDim.x + tidx) * 2 + (count + Hei + Col)%2;


  int index = tidx + tidy * blockDim.x + tidz * blockDim.x * blockDim.y;
  double tmp = b[Hei * NN + Col * N + Row]
  + u[(Hei+1) * N2N + (Col + 1)*N2 + Row ]
  + u[(Hei+1) * N2N + (Col + 1)*N2 + Row + 2]
  + u[(Hei+1) * N2N + Col * N2 + Row + 1]
  + u[(Hei+1) * N2N + (Col + 2) * N2 + Row + 1]
  + u[Hei * N2N + (Col + 1) * N2 + Row + 1]
  + u[(Hei+2) * N2N + (Col + 1)*(N2) + Row + 1];

    __shared__ double uds[blksize]; 
    double res0 =  tmp - 6 * u[(Hei+1) * N2N + (Col + 1) * N2 + Row + 1];
    u[(Hei+1) * N2N + (Col + 1)* N2 + Row + 1] = tmp/6.0;
    uds[index] = res0 * res0;
    __syncthreads();

    for (int idx = blksize/2; idx > 32; idx >>= 1){

      if (index < idx)
        uds[index] += uds[index + idx];
      __syncthreads();

    }

    if (index < 32){
      inReduce1(uds, index);
    }

    if (index == 0)
      res[bidx + bidy * nx + bidz * nx * ny] = uds[0];

}

__global__
void rb_kernel1(double* u, const double* b, double *res, int count){

  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Hei = bidz * blockDim.z + tidz;
  int Col = bidy * blockDim.y + tidy;
  int Row = (bidx * blockDim.x + tidx) * 2 + (count + Hei + Col)%2;


  int index = tidx + tidy * blockDim.x + tidz * blockDim.x * blockDim.y;
  u[(Hei+1) * N2N + (Col + 1)* N2 + Row + 1] =
      (b[Hei * NN + Col * N + Row]
    + u[(Hei+1) * N2N + (Col + 1)*N2 + Row ]
    + u[(Hei+1) * N2N + (Col + 1)*N2 + Row + 2]
    + u[(Hei+1) * N2N + Col * N2 + Row + 1]
    + u[(Hei+1) * N2N + (Col + 2) * N2 + Row + 1]
    + u[Hei * N2N + (Col + 1) * N2 + Row + 1]
    + u[(Hei+2) * N2N + (Col + 1)*(N2) + Row + 1])/6;

}


__device__ void inReduce1(volatile double *uds, int index){
    uds[index] += uds[index + 32];
    uds[index] += uds[index + 16];
    uds[index] += uds[index + 8];
    uds[index] += uds[index + 4];
    uds[index] += uds[index + 2];
    uds[index] += uds[index + 1];
}

