// /*
//   Foundations of Parallel and Distributed Computing, Fall 2021.
//   Instructor: Prof. Chao Yang @ Peking University.
//   Date: 1/11/2021
// */

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
void butterfly_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);


// delete the // to directly run this algorithm directly

//// #define BUFFER_SIZE (4 * 1024 * 1024)
// #define ABS(x) ((x > 0) ? x : (-x))
// #define EPS 1e-5


// double get_walltime() {
// #if 1
//   return MPI_Wtime();
// #else
//   struct timeval tp;
//   gettimeofday(&tp, NULL);
//   return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
// #endif
// }

// void initialize(int rank, float* data, int n) {
//   int i = 0;
//   srand(rank);
//   for (i = 0; i < n; ++i) {
//     data[i] = rand() / (float)RAND_MAX;
//   }
// }

// int result_check(float* a, float* b, int n) {
//   int i = 0;
//   for (i = 0; i < n; ++i) {
//     if (ABS(a[i] - b[i]) > EPS) {
//       return 0;
//     }
//   }
//   return 1;
// }

// float max_error(float* a, float *b, int n){
//   float maxerror = 0;
//   for (int i = 0; i < n; ++i){
//     maxerror = (maxerror > ABS(a[i]-b[i]))? maxerror:ABS(a[i]-b[i]);
//   }
//   return maxerror;
// }
// int main(int argc, char* argv[]) {
//   int rank, comm_size;
//   // batch test
//   int BUFFER_SIZE[4] = {(1024*1024),(4*1024*1024),(16*1024*1024),(64*1024*1024)};
//   MPI_Init(&argc, &argv);
//   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   for (int i = 0; i < 3; i++){
//     float *data = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *base_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *butterfly_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     double time0, time1;
//     double butterfly_time_average=0;
//     double base_time_average=0;
//     double butterfly_error_average=0;
//     int correct_count = 0, correct = 0;



//     // initialization
//     initialize(rank, data, BUFFER_SIZE[i]);
//     // printf("\n rank%d",rank);
//     // for (int j = 0;j<BUFFER_SIZE[i];j++){
//     //   printf(" %f",data[j]);
//     // }
//     // ground true results
//     int maxepoch = (i < 2)?100:20;
//     for (int cnt = 0 ; cnt < maxepoch; cnt ++){
//       double butterfly_time = 0, time_butterfly=0;
//       double base_time = 0, time_base=0;
//       double butterfly_error = 0, error_butterfly=0;
//       correct_count = 0;
//       correct = 0;
//       time0 = get_walltime();
//       MPI_Allreduce(data, base_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_base = get_walltime()-time0;
//       // if (rank == 0){
//       //   printf("\n base rank%d",rank);
//       //   for (int j = 0;j<BUFFER_SIZE[i];j++){
//       //     printf(" %f",base_output[j]);
//       //   }
//       // }
//       // MPI_Barrier(MPI_COMM_WORLD);
//       /* write your codes here */
//       // memcpy(butterfly_output, data, BUFFER_SIZE[i]*sizeof(float));
//       // printf("\n rank%d",rank);
//       // for (int j = 0;j<BUFFER_SIZE[i];j++){
//       //   printf(" %f",butterfly_output[j]);
//       // }
//       for (int j = 0; j < BUFFER_SIZE[i]; j++) butterfly_output[j] = data[j];
//       time1 = get_walltime();
//       butterfly_allreduce(butterfly_output, comm_size, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_butterfly = get_walltime() - time1;
//       // if (rank == 0){
//         // printf("\n rank%d",rank);
//         // for (int j = 0;j<BUFFER_SIZE[i];j++){
//         //   printf(" %f",butterfly_output[j]);
//         // }
//       // }
//       error_butterfly =  max_error(base_output, butterfly_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&error_butterfly, &butterfly_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_butterfly, &butterfly_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_base, &base_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);

//       // check correctness and report results
//       correct = result_check(base_output, butterfly_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

//       // printf("\n rank %d max error %f", rank, error_butterfly);
//       if (rank == 0){
//         butterfly_error_average = (butterfly_error_average * cnt) / (cnt+1) + butterfly_error / (cnt+1);
//         butterfly_time_average = (butterfly_time_average * cnt) / (cnt+1) + butterfly_time / (cnt+1);
//         base_time_average = (base_time_average * cnt) / (cnt+1) + base_time / (cnt+1); 
//       }
//       if (!correct) {
//         printf("Wrong answer on rank %d.\n", rank);
//         break;
//       }
//     }

//     if (rank == 0 && correct_count == comm_size) {
//       printf("Buffer size: %d, comm size: %d\n", BUFFER_SIZE[i], comm_size);
//       printf("Correct results.\n");
//       printf("Your average baseline wall time:%f\n", base_time_average);
//       printf("Your average implementation wall time for butterfly:%f\n", butterfly_time_average);
//       printf("Your average maxerror with basline for butterfly:%e\n", butterfly_error_average);
//     }
//     free(data);
//     free(base_output);
//     free(butterfly_output);
//   }
//   return 0;
// }

void butterfly_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  // this method use butterfly structure for reduce and broadcast
  // when the comm_size is not exponential of 2, the one without partner will recv from the left one.
  int exp=1, stepsize=2;
  int comm_size;
  int rank;
  int dest;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);
  while (exp < comm_size) exp*=2;
  // prepare the slice for send/recv
  // int slicenum = comm_size;
  int minisize = (int)(count/slicenum);
  // printf("\n%d slicenum", slicenum);
  int *seglength = (int*)malloc(slicenum*sizeof(int));
  for (int i = 0;i<slicenum-1;i++) seglength[i] = minisize;
  seglength[slicenum-1] = count-(slicenum-1)*minisize;
  float *tmprecv = (float*)malloc((seglength[slicenum-1])*sizeof(float));
  if (op == MPI_SUM){
    while (stepsize <= exp){
      // match and exchange data 
      if (rank % stepsize < stepsize/2) dest = rank+stepsize/2;
      else dest = rank-stepsize/2;
      if (dest < comm_size){
        for (int j = 0; j < slicenum; j++){
          MPI_Sendrecv(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, dest, 0, 
          tmprecv, seglength[j], MPI_FLOAT, dest, 0, comm, MPI_STATUS_IGNORE);
          for (int i = 0; i < seglength[j];i++)
            recvbuf[j*minisize+i] += tmprecv[i];
        }
      }
      // in this case, not matched, we copy sum from the 1st node of one group
      else if (stepsize >2){
        dest = rank-rank%stepsize;
        if (dest != rank){
          for (int j = 0; j < slicenum; j++){
            MPI_Recv(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, dest, 0, comm, MPI_STATUS_IGNORE);
          }
        }
      }
      if (rank % stepsize == 0 && rank + stepsize > comm_size){
        for (int not_matched_count = rank+stepsize-comm_size; not_matched_count>0; not_matched_count--){
          dest = rank + stepsize/2 - not_matched_count;
          if (dest > rank && dest < comm_size){
            for (int j = 0; j < slicenum; j++){
              MPI_Send(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, dest, 0, comm);
            }
          }
        }
      }
    // no broadcast
    stepsize *= 2;
    }
  }
  //free the memory
  free(tmprecv);
  free(seglength);
  return;
}

