#include "head.h"

__device__ void inReduce1(volatile double *uds, int index);

// simple read/load
__global__
void jacobi_kernel(double* u,const double* u1,const double* b, double *res){

  __shared__ double uds[blksize];
  int bidx = blockIdx.x; 
  int bidy = blockIdx.y;
  int bidz = blockIdx.z;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tidz = threadIdx.z;

  int Row = bidx * blockDim.x + tidx;
  int Col = bidy * blockDim.y + tidy;
  int Hei = bidz * blockDim.z + tidz;

  int index = tidx + tidy * blockDim.x + tidz * blockDim.x * blockDim.y;
  double tmp = b[Hei * NN + Col * N + Row]
  + u1[(Hei+1) * N2N + (Col + 1)*N2 + Row ]
  + u1[(Hei+1) * N2N + (Col + 1)*N2 + Row + 2]
  + u1[(Hei+1) * N2N + Col * N2 + Row + 1]
  + u1[(Hei+1) * N2N + (Col + 2) * N2 + Row + 1]
  + u1[Hei * N2N + (Col + 1) * N2 + Row + 1]
  + u1[(Hei+2) * N2N + (Col + 1)*(N2) + Row + 1];

  double res0 =  tmp - 6 * u1[(Hei+1) * N2N + (Col + 1) * N2 + Row + 1];

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
    // res[bidx + bidy * nx + bidz * nx * ny] = 1;

}

//use pre load data, however, this is slower than simple read/load
__global__ 
void jacobi_kernel2(double *u,const double *u1,const double *b, double *res)
{
  //use shared memory to preload data, however, this is slower than simple load
    int k = blockDim.x * blockIdx.x;
    int j = blockDim.y * blockIdx.y;
    int i = blockDim.z * blockIdx.z;
    int tk = threadIdx.x;
    int tj = threadIdx.y;
    int ti = threadIdx.z;
    __shared__ double edu[ez][ey][ex];
    __shared__ double uds[blksize];
    int index = tk + blockDim.x * tj + blockDim.x * blockDim.y * ti;
    for (int cnt = 0; cnt < (ez * ey * ex) / (blockDim.z * blockDim.y * blockDim.x); cnt++)
    {
        int ntid = index + cnt * (blockDim.z * blockDim.y * blockDim.x);
        if (ntid < ez * ey * ex)
        {
            int nk = ntid % ex;
            int nj = (ntid / ex) % ey;
            int ni = ntid / (ex * ey);
            edu[ni][nj][nk] = u1[(i + ni) * N2N + (j + nj) * N2 + k + nk];
        }
        // __syncthreads();
    }
    __syncthreads();
    k = blockDim.x * blockIdx.x + threadIdx.x;
    j = blockDim.y * blockIdx.y + threadIdx.y;
    i = blockDim.z * blockIdx.z + threadIdx.z;
    double tmp = b[i * N * N + j * N + k] 
          + edu[ti][tj + 1][tk + 1] 
          + edu[ti + 1][tj][tk + 1] 
          + edu[ti + 1][tj + 1][tk] 
          + edu[ti + 1][tj + 1][tk + 2] 
          + edu[ti + 1][tj + 2][tk + 1] 
          + edu[ti + 2][tj + 1][tk + 1];

    u[(i + 1) * N2N + (j + 1) * N2 + k + 1] = tmp / 6.0;
    double res0 = tmp - 6.0 * edu[ti+1][tj+1][tk+1];
    res[index] = res0 * res0;
    // Adjusted from kernel5 in res.cu
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
    res[blockIdx.x + blockIdx.y * nx + blockIdx.z * nx * ny] = uds[0];
    // res[bidx + bidy * nx + bidz * nx * ny] = 1;

}

__device__ void inReduce1(volatile double *uds, int index){
    uds[index] += uds[index + 32];
    uds[index] += uds[index + 16];
    uds[index] += uds[index + 8];
    uds[index] += uds[index + 4];
    uds[index] += uds[index + 2];
    uds[index] += uds[index + 1];
}

