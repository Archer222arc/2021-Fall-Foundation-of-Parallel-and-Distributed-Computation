#include "head.h"

// the first step
__global__
void initial_step(double* u, const double* b){
  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Hei = bidz * blockDim.z + tidz;
  int Col = bidy * blockDim.y + tidy;
  int Row = (bidx * blockDim.x + tidx) * 2;
  __syncthreads();

  u[(Hei+1) * N2N + (Col + 1)* N2 + Row + 1] = b[Hei * NN + Col * N + Row] /6.0;

}
