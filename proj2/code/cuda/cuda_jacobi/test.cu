#include "jacobi.h"
#include "head.h"
#include "cuda.h"
// __host__
int main() {
  double * u = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
  double * u_exact = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
  double * b = (double *)malloc(sizeof(double) * N * N * N);
  init_sol(b, u_exact, u);

  double *d_u1, *d_u2;
  double *d_b;
  double *res, *res1;
  double normr0 = residual_norm(u, b);
  double normr = normr0;
  double *d_normr;
  int tsteps = MAXITER;
  int k;

  size_t size_of_double = sizeof(double);
  size_t size_u = size_of_double * (N+2)*(N+2)*(N+2);
  size_t size_b = size_of_double * N * N * N;
  size_t size_res = size_of_double * nx * ny * nz;

  cudaMalloc((double**)&d_u1, size_u);
  cudaMalloc((double**)&d_u2, size_u);
  cudaMalloc((double**)&d_b, size_b);
  cudaMalloc((double**)&d_normr,size_of_double);
  cudaMalloc((double**)&res, size_res);
  cudaMalloc((double**)&res1, size_of_double * rstep);


  cudaMemcpy(d_u1, u, size_u, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u2, u, size_u, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
  cudaMemset(res, 0, size_res);
  cudaMemset(res1, 0, size_of_double * rstep);
  cudaMemset(d_normr, 0, size_of_double);



  dim3 grid_dim(N/lx, N/ly, N/lz);
  dim3 block_dim(lx, ly, lz);

  double kernel_time = 0.0;
  long long gpu_start_time = start_timer();
  for (k = 0; k < MAXITER; k++)
  {
      cudaEvent_t start, stop;
      float elapsed_time = 0.0;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
      
      if (k == 0){
        initial_step<<<grid_dim,block_dim, blksize*size_of_double>>>(d_u2, d_b);
      }
      // 主要迭代
      else if (k%2 == 1){
        jacobi_kernel<<<grid_dim, block_dim, blksize*size_of_double>>>(d_u1, d_u2, d_b, res);
      }
      else{ 
        jacobi_kernel<<<grid_dim, block_dim, blksize*size_of_double>>>(d_u2, d_u1, d_b, res);
      }
      double tmp[gridsize];

      // 残差规约
      if (k > 0){
        reduce1<<< rstep , rsize, rsize * size_of_double>>>(res, res1);
        reduce2<<< 1, rstep, rstep * size_of_double>>>(res1, d_normr); 
      }


      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed_time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      kernel_time += elapsed_time / 1000;
      if (k > 0){

        cudaMemcpy(&normr, d_normr, size_of_double, cudaMemcpyDeviceToHost);
        normr = sqrt(normr);
        printf("Iteration %d, normr/normr0=%g, time cost=%g\n", k, normr / normr0, elapsed_time / 1000);
      }
      if (normr < RTOL * normr0 && k >= 33)
      {
        float elapsed_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        if (k%2 == 0)
          residual<<<grid_dim, block_dim, blksize*size_of_double>>>(d_u2, d_b, res);
        else 
          residual<<<grid_dim, block_dim, blksize*size_of_double>>>(d_u1, d_b, res);

        dim3 grid_dim(N/lx, N/ly, N/lz);
        dim3 block_dim(lx, ly, lz);

        reduce1<<< rstep , rsize, rsize * size_of_double>>>(res, res1);
        reduce2<<< 1, rstep, rstep * size_of_double>>>(res1, d_normr); 
        cudaMemcpy(&normr, d_normr, size_of_double, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        normr = sqrt(normr);
        kernel_time += elapsed_time / 1000;
        printf("Iteration %d, normr/normr0=%g, time cost=%g\n", k+1, normr / normr0 , elapsed_time / 1000);
        tsteps = k;
        printf("Converged with %d iterations.\n", tsteps);
        break;
      }
  }
  long long gpu_time = stop_timer(gpu_start_time, "total time");
  long long residual_norm_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + (N * N * N)) * tsteps;
  long long gs_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + 2 * (N * N * N)) * tsteps;

  printf("kernel time: %g\n", kernel_time);

  long long total_bytes = residual_norm_bytes + gs_bytes;
  double bandwidth = total_bytes / gpu_time;

  printf("total bandwidth: %g GB/s\n", bandwidth / (double)(1 << 30));
  if (k%2 == 1){
    cudaMemcpy(u, d_u1, size_u, cudaMemcpyDeviceToHost);
  }
  else {
    cudaMemcpy(u, d_u2, size_u, cudaMemcpyDeviceToHost);
  }
  cudaFree(d_u1);
  cudaFree(d_u2);
  cudaFree(d_b);
  cudaFree(d_normr);

  double final_normr = residual_norm(u, b); // Please ensure that this residual_norm is exact.
  printf("Final residual norm: %g\n", final_normr);
  printf("|r_n|/|r_0| = %g\n", final_normr / normr0);
  double relative_err = error(u, u_exact);
  printf("relative error: %g\n", relative_err);
  free(u);
  free(u_exact);
  free(b);
  return 0;
}