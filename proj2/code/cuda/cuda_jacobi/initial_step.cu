#include "head.h"
#include "cuda.h"

// #include <cuda.h>
__global__
void initial_step(double* u, const double* b){
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int Row = bx * lx + tx;
  int Col = by * ly + ty;
  int Hei = bz * lz + tz;
  __syncthreads();
  u[(Row+1) * N2N + (Col + 1)*(N+2) + Hei + 1] = b[Row * NN + Col * N + Hei]/6;

}
