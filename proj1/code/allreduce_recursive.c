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

void recursive_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

// delete the comment in order to run this algorithm directly with slicenum = commsize


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
//   // int BUFFER_SIZE[4] = {(1024*1024),(4*1024*1024),(16*1024*1024),(64*1024*1024)};
//   int BUFFER_SIZE[4] = {16,4,4,4};
//   MPI_Init(&argc, &argv);
//   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   for (int i = 0; i < 1; i++){
//     float *data = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *base_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *recursive_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     double time0, time1;
//     double recursive_time_average=0;
//     double base_time_average=0;
//     float recursive_error_average=0;
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
//       double base_time = 0, time_base=0;
//       double recursive_time = 0, time_recursive=0;
//       float recursive_error = 0, error_recursive=0;
//       correct_count = 0;
//       correct = 0;
//       time0 = get_walltime();
//       MPI_Allreduce(data, base_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_base = get_walltime()-time0;
//       // MPI_Barrier(MPI_COMM_WORLD);
//       /* write your codes here */
//       // for (int j = 0; j < BUFFER_SIZE[i]; j++) recursive_output[j] = data[j];
//       memcpy(recursive_output, data, BUFFER_SIZE[i]*sizeof(float));

//       time1 = get_walltime();
//       recursive_allreduce(recursive_output, comm_size, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_recursive = get_walltime() - time1;

//       error_recursive =  max_error(base_output, recursive_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&error_recursive, &recursive_error, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_recursive, &recursive_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_base, &base_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);

//       // check correctness and report results
//       correct = result_check(base_output, recursive_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

//       if (rank == 0){
//         recursive_error_average = (recursive_error_average * cnt) / (cnt+1) + recursive_error / (cnt+1);
//         recursive_time_average = (recursive_time_average * cnt) / (cnt+1) + recursive_time / (cnt+1);
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
//       printf("Your average implementation wall time for recursive:%f\n", recursive_time_average);
//       printf("Your average maxerror with basline for recursive:%e\n", recursive_error_average);
//     }
//     free(data);
//     free(base_output);
//     free(recursive_output);
//   }
//   return 0;
// }
void recursive_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  // this method use binaryrecursive-like structure for reduce and broadcast
  // when the comm_size is not exponential of 2, we omit those lacked nodes.
  int exp=1, res;
  int comm_size, current_size;
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  // prepare the slice for send/recv
  // int slicenum = comm_size;
  int minisize = (int)(count/slicenum);
  int *seglength = (int*)malloc(slicenum*sizeof(int));
  for (int i = 0;i<slicenum-1;i++) seglength[i] = minisize;
  seglength[slicenum-1] = count-(slicenum-1)*minisize;
  float *tmprecv = (float*)malloc((seglength[slicenum-1])*sizeof(float));

  //find the minimal exponential of 2 not less than comm_size.
  while (exp < comm_size) exp *=2;  
  //the rest number of process in last epoch
  res = comm_size-exp/2;
  //send   rank --> rank- current_size/2, until only 1 process left.
  current_size = exp;
  // use recursive structre
  if (op == MPI_SUM){
    for (current_size = exp; current_size > 1; current_size /= 2){
        if (rank >= current_size) break;
        if (rank >= current_size/2){
          for (int j = 0; j < slicenum;j++){
            MPI_Send(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, rank-current_size/2, rank*slicenum+j, comm);
          }
        }
        else if (rank + current_size/2 < comm_size){
          for (int j = 0; j < slicenum; j++){
            MPI_Recv(tmprecv, seglength[j], MPI_FLOAT, rank+current_size/2, (rank+current_size/2)*slicenum+j, comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < seglength[j]; i++){ 
              recvbuf[j*minisize+i]= tmprecv[i]+recvbuf[j*minisize+i];
            }
          }
        }
      }
    }
    exp /= 2;
    current_size = 1;
    // broadcast it should be cautious when comm_size of not exponential of 2.
    while (current_size < exp){
      if (rank < current_size) MPI_Send(recvbuf, count, MPI_FLOAT, rank+current_size, 0, comm);
      else if (rank < 2*current_size) MPI_Recv(recvbuf, count, MPI_FLOAT, rank-current_size, 0, comm, MPI_STATUS_IGNORE);
      current_size *= 2;
    }
    if (rank < res){
      for (int j = 0; j < slicenum; j++)
        MPI_Send(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, rank+exp, 0, comm);
    }
    else if ((rank >= exp) && (rank < comm_size)){
      for (int j = 0; j < slicenum; j++)
        MPI_Recv(&recvbuf[j*minisize], seglength[j], MPI_FLOAT, rank-exp, 0, comm, MPI_STATUS_IGNORE);
    }
    //释放内存
    free(tmprecv);
    free(seglength);
  return;
}

