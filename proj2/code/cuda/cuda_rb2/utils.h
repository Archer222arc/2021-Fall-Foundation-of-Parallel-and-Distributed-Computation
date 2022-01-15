#include "constant.h"
#include <cuda.h>

// __device__ void inReduce(volatile double *uds, int index);


// __device__ void inReduce(volatile double *uds, int index){
//     uds[index] += uds[index + 32];
//     uds[index] += uds[index + 16];
//     uds[index] += uds[index + 8];
//     uds[index] += uds[index + 4];
//     uds[index] += uds[index + 2];
//     uds[index] += uds[index + 1];
// }
