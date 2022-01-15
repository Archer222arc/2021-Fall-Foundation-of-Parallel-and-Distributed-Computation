/*
  Foundations of Parallel and Distributed Computing, Fall 2021.
  Instructor: Prof. Chao Yang @ Peking University.
  Date: 1/11/2021
*/

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>




void star_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

// delete the comment to run algorithm directly

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
//   int BUFFER_SIZE[4] = {(1024*1024),(4*1024*1024),(16*1024*1024),(64*1024*1024)};
//   // int BUFFER_SIZE[4] = {16,4,4,4};
//   MPI_Init(&argc, &argv);
//   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   for (int i = 0; i < 3; i++){
//     float *data = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *base_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *star_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     double time0, time1;
//     double star_time_average=0;
//     double base_time_average=0;
//     double star_error_average=0;
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
//       double star_time = 0, time_star=0;
//       double base_time = 0, time_base=0;
//       double star_error = 0, error_star=0;
//       correct_count = 0;
//       correct = 0;
//       time0 = get_walltime();
//       MPI_Allreduce(data, base_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_base = get_walltime()-time0;

//       // MPI_Barrier(MPI_COMM_WORLD);
//       /* write your codes here */
//       // memcpy(star_output, data, BUFFER_SIZE[i]*sizeof(float));

//       for (int j = 0; j < BUFFER_SIZE[i]; j++) star_output[j] = data[j];
//       time1 = get_walltime();
//       star_allreduce(star_output, comm_size, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_star = get_walltime() - time1;

//       error_star =  max_error(base_output, star_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&error_star, &star_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_star, &star_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_base, &base_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);

//       // check correctness and report results
//       correct = result_check(base_output, star_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

//       // printf("\n rank %d max error %f", rank, error_star);
//       if (rank == 0){
//         star_error_average = (star_error_average * cnt) / (cnt+1) + star_error / (cnt+1);
//         star_time_average = (star_time_average * cnt) / (cnt+1) + star_time / (cnt+1);
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
//       printf("Your average implementation wall time for star:%f\n", star_time_average);
//       printf("Your average maxerror with basline for star:%e\n", star_error_average);
//     }
//     free(data);
//     free(base_output);
//     free(star_output);
//   }
//   return 0;
// }

void star_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  // this method use ring structure for reduce and broadcast
  // each epoch, the Nth node will send the (N-j)th slice to N+1, and recv the (N-j-1)th slice from N-1
  int comm_size;
  // MPI_Request req;
  int rank;
  // MPI_Request req;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);
  // prepare the slice for send/recv
  // int slicenum = comm_size;
  int minisize = (int)(count/slicenum);
  int *seglength = (int*)malloc(slicenum*sizeof(int));
  for (int i = 0;i<slicenum-1;i++) seglength[i] = minisize;
  seglength[slicenum-1] = count-(slicenum-1)*minisize;
  float *tmprecv = (float*)malloc((seglength[slicenum-1])*sizeof(float));

  if (op == MPI_SUM){
    // send all date to node 0 and compute 
    for (int j = 0; j < slicenum; j++){
      if (rank != 0){
        MPI_Send(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, 0, rank*slicenum+j, comm);
      }
      else {
        for (int k = 1; k < comm_size; k++){
          MPI_Recv(tmprecv, seglength[j], MPI_FLOAT, k, k*slicenum+j, comm, MPI_STATUS_IGNORE);
          // MPI_Irecv(tmprecv, seglength[j], MPI_FLOAT, k, k*slicenum+j, comm, &req);
          for (int i = 0; i < seglength[j]; i++)
            recvbuf[j*minisize+i] += tmprecv[i];
        }
      }
    }
  }
  if (rank == 0){
  // broadcast
    for (int k = 1; k < comm_size; k++){
      for (int j = 0; j < slicenum; j++){
        MPI_Send(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, k, k*slicenum+j, comm);
      }
    }
  }
  if (rank > 0)
    for (int j = 0; j < slicenum; j++){
      MPI_Recv(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, 0, rank*slicenum+j, comm, MPI_STATUS_IGNORE);
    }
  //free the memory
  free(tmprecv);
  free(seglength);
  return;
}

