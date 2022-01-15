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
#include "allreduce.h"
// #define BUFFER_SIZE (4 * 1024 * 1024)
// #define BUFFER_SIZE (8)
#define ABS(x) ((x > 0) ? x : (-x))
#define EPS 1e-5


double get_walltime() {
#if 1
  return MPI_Wtime();
#else
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
#endif
}

void initialize(int rank, float* data, int n) {
  int i = 0;
  srand(rank);
  for (i = 0; i < n; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

int result_check(float* a, float* b, int n) {
  int i = 0;
  for (i = 0; i < n; ++i) {
    if (ABS(a[i] - b[i]) > EPS) {
      return 0;
    }
  }
  return 1;
}

float max_error(float* a, float *b, int n){
  float maxerror = 0;
  for (int i = 0; i < n; ++i){
    maxerror = (maxerror > ABS(a[i]-b[i]))? maxerror:ABS(a[i]-b[i]);
  }
  return maxerror;
}

int main(int argc, char* argv[]) {
  int rank, comm_size;
  int BUFFER_SIZE[4] = {(1024*1024),(4*1024*1024),(8*1024*1024),(64*1024*1024)};
  // int BUFFER_SIZE[4] = {16,4,4,4};
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int slicenum = comm_size*4;
  for (int i = 0; i < 3; i++){
    float *data = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *base_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *butterfly_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *recursive_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *ring_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *circle_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *star_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *obutterfly_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *orecursive_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *ocircle_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));
    float *ostar_output = (float*)malloc(BUFFER_SIZE[i]*sizeof(float));

    double time0, time1;
    double butterfly_time_average=0;
    double recursive_time_average=0;    
    double ring_time_average=0;    
    double circle_time_average=0;    
    double star_time_average=0;
    double obutterfly_time_average=0;
    double orecursive_time_average=0;    
    double ocircle_time_average=0;    
    double ostar_time_average=0;
    
    double base_time_average=0;
    double butterfly_error_average=0;    
    double recursive_error_average=0;    
    double ring_error_average=0;    
    double circle_error_average=0;    
    double star_error_average=0;
    double obutterfly_error_average=0;    
    double orecursive_error_average=0;    
    double ocircle_error_average=0;    
    double ostar_error_average=0;
    int correct_count = 0, correct = 0;

    int maxepoch = (i < 2)?100:30;
    if (comm_size > 8 && i > 2) maxepoch =1;


    // initialization
    initialize(rank, data, BUFFER_SIZE[i]);
    // printf("\n rank%d",rank);
    // for (int j = 0;j<BUFFER_SIZE[i];j++){
    //   printf(" %f",data[j]);
    // }
    // ground true results
    // int maxepoch = 1;
    for (int cnt = 0 ; cnt < maxepoch; cnt ++){
      double base_time = 0, time_base=0;
      double butterfly_time = 0, time_butterfly=0;
      double butterfly_error = 0, error_butterfly=0;
      double recursive_time = 0, time_recursive=0;
      double recursive_error = 0, error_recursive=0;
      double ring_error = 0, error_ring=0;
      double ring_time = 0, time_ring=0;
      double circle_time = 0, time_circle=0;
      double circle_error = 0, error_circle=0;
      double star_time = 0, time_star=0;
      double star_error = 0, error_star=0;
      double obutterfly_time = 0, otime_butterfly=0;
      double obutterfly_error = 0, oerror_butterfly=0;
      double orecursive_time = 0, otime_recursive=0;
      double orecursive_error = 0, oerror_recursive=0;
      double ocircle_time = 0, otime_circle=0;
      double ocircle_error = 0, oerror_circle=0;
      double ostar_time = 0, otime_star=0;
      double ostar_error = 0, oerror_star=0;
      correct_count = 0;
      correct = 0;
      time0 = get_walltime();
      MPI_Allreduce(data, base_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time_base = get_walltime()-time0;

      // for (int j = 0; j < BUFFER_SIZE[i]; j++) butterfly_output[j] = data[j];
      memcpy(butterfly_output,data,BUFFER_SIZE[i]*sizeof(float));
      time1 = get_walltime();
      butterfly_allreduce(butterfly_output, slicenum, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time_butterfly = get_walltime() - time1;

      // for (int j = 0; j < BUFFER_SIZE[i]; j++) obutterfly_output[j] = data[j];
      memcpy(obutterfly_output,data,BUFFER_SIZE[i]*sizeof(float));
      time1 = get_walltime();
      butterfly_allreduce(obutterfly_output, 1, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      otime_butterfly = get_walltime() - time1;

      memcpy(recursive_output,data,BUFFER_SIZE[i]*sizeof(float));
      // for (int j = 0; j < BUFFER_SIZE[i]; j++) recursive_output[j] = data[j];
      time1 = get_walltime();
      recursive_allreduce(recursive_output, slicenum, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time_recursive = get_walltime() - time1;

      // for (int j = 0; j < BUFFER_SIZE[i]; j++) orecursive_output[j] = data[j];
      memcpy(orecursive_output,data,BUFFER_SIZE[i]*sizeof(float));
      time1 = get_walltime();
      recursive_allreduce(orecursive_output, 1, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      otime_recursive = get_walltime() - time1;

      // for (int j = 0; j < BUFFER_SIZE[i]; j++) ring_output[j] = data[j];
      memcpy(ring_output,data,BUFFER_SIZE[i]*sizeof(float));
      time1 = get_walltime();
      ring_allreduce(ring_output, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time_ring = get_walltime() - time1;

      // for (int j = 0; j < BUFFER_SIZE[i]; j++) circle_output[j] = data[j];
      memcpy(circle_output,data,BUFFER_SIZE[i]*sizeof(float));
      time1 = get_walltime();
      circle_allreduce(circle_output, slicenum, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time_circle = get_walltime() - time1;

      // for (int j = 0; j < BUFFER_SIZE[i]; j++) ocircle_output[j] = data[j];
      memcpy(ocircle_output,data,BUFFER_SIZE[i]*sizeof(float));
      time1 = get_walltime();
      circle_allreduce(ocircle_output, 1, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      otime_circle = get_walltime() - time1;

      memcpy(star_output,data,BUFFER_SIZE[i]*sizeof(float));
      // for (int j = 0; j < BUFFER_SIZE[i]; j++) star_output[j] = data[j];
      time1 = get_walltime();
      star_allreduce(star_output, slicenum, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      time_star = get_walltime() - time1;

      memcpy(ostar_output,data,BUFFER_SIZE[i]*sizeof(float));
      // for (int j = 0; j < BUFFER_SIZE[i]; j++) ostar_output[j] = data[j];
      time1 = get_walltime();
      star_allreduce(ostar_output, 1, BUFFER_SIZE[i], MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      otime_star = get_walltime() - time1;

      error_butterfly =  max_error(base_output, butterfly_output, BUFFER_SIZE[i]);
      error_recursive =  max_error(base_output, recursive_output, BUFFER_SIZE[i]);
      error_ring =  max_error(base_output, ring_output, BUFFER_SIZE[i]);
      error_circle =  max_error(base_output, circle_output, BUFFER_SIZE[i]);
      error_star =  max_error(base_output, star_output, BUFFER_SIZE[i]);
      oerror_butterfly =  max_error(base_output, obutterfly_output, BUFFER_SIZE[i]);
      oerror_recursive =  max_error(base_output, orecursive_output, BUFFER_SIZE[i]);
      oerror_circle =  max_error(base_output, ocircle_output, BUFFER_SIZE[i]);
      oerror_star =  max_error(base_output, ostar_output, BUFFER_SIZE[i]);
      // if (rank == 0){
      //   printf("\nepoch %d maxerror = %e\n", cnt, error_recursive);
      //   for (int j = 0; j < BUFFER_SIZE[i];j++){
      //     printf(" %f",recursive_output[j]);
      //   }
      //   printf("\nepoch %d base_output", cnt);
      //   for (int j = 0; j < BUFFER_SIZE[i];j++){
      //     printf(" %f",base_output[j]);
      //   }
      // }

      MPI_Reduce(&error_star, &star_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_star, &star_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&error_circle, &circle_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_circle, &circle_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&error_ring, &ring_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_ring, &ring_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&error_recursive, &recursive_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_recursive, &recursive_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&error_butterfly, &butterfly_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&time_butterfly, &butterfly_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      MPI_Reduce(&oerror_star, &ostar_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&otime_star, &ostar_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&oerror_circle, &ocircle_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&otime_circle, &ocircle_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&oerror_recursive, &orecursive_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&otime_recursive, &orecursive_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&oerror_butterfly, &obutterfly_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&otime_butterfly, &obutterfly_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      MPI_Reduce(&time_base, &base_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      // check correctness and report results
      correct = result_check(base_output, butterfly_output, BUFFER_SIZE[i]);
      MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

      // printf("\n rank %d max error %f", rank, error_butterfly);
      if (rank == 0){
        butterfly_error_average = (butterfly_error_average * cnt) / (cnt+1) + butterfly_error / (cnt+1);
        butterfly_time_average = (butterfly_time_average * cnt) / (cnt+1) + butterfly_time / (cnt+1);
        recursive_error_average = (recursive_error_average * cnt) / (cnt+1) + recursive_error / (cnt+1);
        recursive_time_average = (recursive_time_average * cnt) / (cnt+1) + recursive_time / (cnt+1);
        ring_error_average = (ring_error_average * cnt) / (cnt+1) + ring_error / (cnt+1);
        ring_time_average = (ring_time_average * cnt) / (cnt+1) + ring_time / (cnt+1);
        star_error_average = (star_error_average * cnt) / (cnt+1) + star_error / (cnt+1);
        star_time_average = (star_time_average * cnt) / (cnt+1) + star_time / (cnt+1);
        circle_error_average = (circle_error_average * cnt) / (cnt+1) + circle_error / (cnt+1);
        circle_time_average = (circle_time_average * cnt) / (cnt+1) + circle_time / (cnt+1);
        obutterfly_error_average = (obutterfly_error_average * cnt) / (cnt+1) + obutterfly_error / (cnt+1);
        obutterfly_time_average = (obutterfly_time_average * cnt) / (cnt+1) + obutterfly_time / (cnt+1);
        orecursive_error_average = (orecursive_error_average * cnt) / (cnt+1) + orecursive_error / (cnt+1);
        orecursive_time_average = (orecursive_time_average * cnt) / (cnt+1) + orecursive_time / (cnt+1);
        ostar_error_average = (ostar_error_average * cnt) / (cnt+1) + ostar_error / (cnt+1);
        ostar_time_average = (ostar_time_average * cnt) / (cnt+1) + ostar_time / (cnt+1);
        ocircle_error_average = (ocircle_error_average * cnt) / (cnt+1) + ocircle_error / (cnt+1);
        ocircle_time_average = (ocircle_time_average * cnt) / (cnt+1) + ocircle_time / (cnt+1);
        base_time_average = (base_time_average * cnt) / (cnt+1) + base_time / (cnt+1); 
      }
      if (!correct) {
        printf("Wrong answer on rank %d.\n", rank);
        break;
      }
    }

    if (rank == 0 && correct_count == comm_size) {
      printf("\nBuffer size: %d, comm size: %d after %d epoch\n", BUFFER_SIZE[i], comm_size, maxepoch);
      printf("Correct results.\n");
      printf("Average baseline wall time:%f\n", base_time_average);
      printf("Average implementation wall time for recursive:%f\n", recursive_time_average);
      printf("Average implementation wall time for butterfly:%f\n", butterfly_time_average);
      printf("Average implementation wall time for ring:%f\n", ring_time_average);
      printf("Average implementation wall time for circle:%f\n", circle_time_average);
      printf("Average implementation wall time for star:%f\n", star_time_average);
      printf("Average maxerror with basline for recursive:%e\n", recursive_error_average);
      printf("Average maxerror with basline for butterfly:%e\n", butterfly_error_average);
      printf("Average maxerror with basline for ring:%e\n", ring_error_average);
      printf("Average maxerror with basline for circle:%e\n", circle_error_average);
      printf("Average maxerror with basline for star:%e\n", star_error_average);
      printf("Average implementation wall time for outplace recursive:%f\n", orecursive_time_average);
      printf("Average implementation wall time for outplace butterfly:%f\n", obutterfly_time_average);
      printf("Average implementation wall time for outplace circle:%f\n", ocircle_time_average);
      printf("Average implementation wall time for outplace star:%f\n", ostar_time_average);
      printf("Average maxerror with basline for outplace recursive:%e\n", orecursive_error_average);
      printf("Average maxerror with basline for outplace butterfly:%e\n", obutterfly_error_average);
      printf("Average maxerror with basline for outplace circle:%e\n", ocircle_error_average);
      printf("Average maxerror with basline for outplace star:%e\n", ostar_error_average);

    }
    free(data);
    free(base_output);
    free(butterfly_output);
  }
  return 0;
}
