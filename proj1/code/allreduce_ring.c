#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
void ring_allreduce(float *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);


// delete the comment to test algorithm directly with commsize = slicenum
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
//   int BUFFER_SIZE[4] = {(1024*1024),(4*1024*1024),(16*1024*1024),(64*1024*1024)};
//   MPI_Init(&argc, &argv);
//   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   for (int i = 0; i < 3; i++){
//     float *data = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *base_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     float *ring_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
//     double time0, time1;
//     double ring_time_average=0;
//     double base_time_average=0;
//     double ring_error_average=0;
//     int correct_count = 0, correct = 0;
//     // initialization
//     initialize(rank, data, BUFFER_SIZE[i]);

//     // ground true results
//     int maxepoch = (i < 2)?100:30;
//     for (int cnt = 0 ; cnt < maxepoch; cnt ++){
//       double ring_error = 0, error_ring=0;
//       double ring_time = 0, time_ring=0;
//       double base_time = 0, time_base=0;

//       time0 = get_walltime();
//       MPI_Allreduce(data, base_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_base = get_walltime()-time0;

//       /* write your codes here */

//       for (int j = 0; j < BUFFER_SIZE[i]; j++) ring_output[j] = data[j];
//       time1 = get_walltime();
//       ring_allreduce(ring_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
//                     MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);
//       time_ring = get_walltime() - time1;


//       error_ring =  max_error(base_output, ring_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&error_ring, &ring_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_ring, &ring_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Reduce(&time_base, &base_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//       MPI_Barrier(MPI_COMM_WORLD);

//       // check correctness and report results
//       correct = result_check(base_output, ring_output, BUFFER_SIZE[i]);
//       MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

//       if (rank == 0){
//         ring_error_average = ring_error_average * cnt / (cnt+1) + ring_error / (cnt+1);
//         ring_time_average = ring_time_average * cnt / (cnt+1) + ring_time / (cnt+1);
//         base_time_average = base_time_average * cnt / (cnt+1) + base_time/ (cnt+1); 
//       }
//     }

//     if (!correct) {
//       printf("Wrong answer on rank %d.\n", rank);
//     }
//     if (rank == 0 && correct_count == comm_size) {
//       printf("Buffer size: %d, comm size: %d\n", BUFFER_SIZE[i], comm_size);
//       printf("Correct results.\n");
//       printf("Your average baseline wall time:%f\n", base_time_average);
//       printf("Your average implementation wall time for ring:%f\n", ring_time_average);
//       printf("Your average maxerror with basline for ring:%e\n", ring_error_average);
//     }
//     free(data);
//     free(base_output);
//     free(ring_output);
//   }
//   return 0;
// }

void ring_allreduce(float *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  // this method use ring structure for reduce and broadcast
  // each epoch, the Nth node will send the (N-j)th slice to N+1, and recv the (N-j-1)th slice from N-1
  int comm_size;
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);
  // prepare the slice for send/recv
  int minisize = (int)(count/comm_size);
  int *seglength = (int*)malloc(comm_size*sizeof(int));
  for (int i = 0;i<comm_size-1;i++) seglength[i] = minisize;
  seglength[comm_size-1] = count-(comm_size-1)*minisize;
  float *tmprecv = (float*)malloc((seglength[comm_size-1])*sizeof(float));
  if (op == MPI_SUM){
  // main cycle, each cycle, recv/send one slice to neighbored nodes.

    for (int j = 0; j < comm_size-1; j++){
      MPI_Sendrecv(&recvbuf[minisize*((-j+rank+comm_size)%comm_size)], seglength[(-j+rank+comm_size)%comm_size], MPI_FLOAT, (rank+1)%comm_size, rank*comm_size+j,
                   tmprecv, seglength[(-j+rank-1+comm_size)%comm_size], MPI_FLOAT, (rank-1+comm_size)%comm_size, ((rank-1+comm_size)%comm_size)*comm_size+j, comm, MPI_STATUS_IGNORE);
      for (int i = 0; i < seglength[(-j-1+rank+comm_size)%comm_size]; i++){
        recvbuf[minisize*((-j-1+rank+comm_size)%comm_size)+i] += tmprecv[i];
      }
    }

// broadcast , epoch 2 in report
    for (int j =0; j < comm_size-1;j++){
      MPI_Sendrecv(&recvbuf[minisize*((rank+1+comm_size-j)%comm_size)], seglength[(rank+1+comm_size-j)%comm_size], MPI_FLOAT, (rank+1)%comm_size, rank*comm_size+j,
                  &recvbuf[minisize*((-j+rank+comm_size)%comm_size)], seglength[(-j+rank+comm_size)%comm_size], MPI_FLOAT, (rank-1+comm_size)%comm_size, ((rank-1+comm_size)%comm_size)*comm_size+j, comm, MPI_STATUS_IGNORE);
    }
  }
  //free the memory
    free(tmprecv);
    free(seglength);
  return;
}

