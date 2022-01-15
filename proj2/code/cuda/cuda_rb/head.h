#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>

#include "utils.h"
#include "constant.h"


// __device__ void inReduce(volatile double *uds, int index){
//     uds[index] += uds[index + 32];
//     uds[index] += uds[index + 16];
//     uds[index] += uds[index + 8];
//     uds[index] += uds[index + 4];
//     uds[index] += uds[index + 2];
//     uds[index] += uds[index + 1];
// }

