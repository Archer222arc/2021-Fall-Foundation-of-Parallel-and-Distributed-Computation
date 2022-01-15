#include "head.h"
#include <cuda.h>


// The first step
__global__
void initial_step(double* u, const double* b, double *res){
  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Hei = bidz * blockDim.z + tidz;
  int Col = bidy * blockDim.y + tidy;
  int Row = bidx * blockDim.x + tidx;
  __syncthreads();
  double tmp = b[Hei * NN/2 + Col * N/2 + Row];
  u[(Hei+1) * N22N + (Col + 1)* N22 + Row + 1] = tmp /6.0;

}
