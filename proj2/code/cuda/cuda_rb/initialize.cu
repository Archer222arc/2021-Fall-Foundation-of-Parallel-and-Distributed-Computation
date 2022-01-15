#include "head.h"
#include "cuda.h"


// (d_u1, d_b1, d_u, d_b, 1)
// copy date to mini grid
#include <cuda.h>
__global__
void initialize(double* du, double* db, const double* u, const double* b, int count){

  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Hei = bidz * blockDim.z + tidz;
  int Col = bidy * blockDim.y + tidy;
  int Row = bidx * blockDim.x + tidx;
  int Row1 = (bidx * blockDim.x + tidx) * 2 + (Hei+Col+count)%2;

  du[(Hei+1) * N22N + (Col + 1)* N22 + Row + 1] = u[(Hei+1) * N2N + (Col + 1)* N2 + Row1 + 1];
  db[Hei * NN/2 + Col * N/2 + Row] = b[Hei * NN + Col * N + Row1];

}
