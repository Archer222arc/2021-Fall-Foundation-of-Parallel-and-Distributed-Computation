#include "head.h"


//copy minigrid to the big grid
__global__
void finalize(const double* du, double* u, int count){
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

  u[(Hei+1) * N2N + (Col + 1)* N2 + Row1 + 1] = du[(Hei+1) * N22N + (Col + 1)* N22 + Row + 1];

}
