#include "rb.h"
#include "head.h"
#include "cuda.h"
int main() {
  double * u = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
  double * u_exact = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
  double * b = (double *)malloc(sizeof(double) * N * N * N);
  init_sol(b, u_exact, u);

  double *d_u;
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
  size_t size_res = size_of_double * gridsize;



  cudaMalloc((double**)&d_u, size_u);
  cudaMalloc((double**)&d_b, size_b);
  cudaMalloc((double**)&d_normr,size_of_double);
  cudaMalloc((double**)&res, size_res);
  cudaMalloc((double**)&res1, size_of_double * rstep);



  cudaMemcpy(d_u, u, size_u, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
  cudaMemset(res, 0, size_res);
  cudaMemset(res1, 0, size_of_double * rstep);
  cudaMemset(d_normr, 0, size_of_double);



  dim3 grid_dim(nx, ny, nz);
  dim3 block_dim(lx, ly, lz);

  double kernel_time = 0.0;
  long long gpu_start_time = start_timer();
  for (k = 0; k < MAXITER; k++)
  {
  
      cudaEvent_t start, stop;
      float elapsed_time = 0.0;
      float time_r = 0.0;
      float time_k = 0.0;
      if (k == 0){
        printf("Iteration %d, normr/normr0=%g", k, normr / normr0);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
      
        initial_step<<<grid_dim,block_dim, blksize*size_of_double>>>(d_u, d_b);
        rb_kernel1<<<grid_dim, block_dim>>>(d_u, d_b, res, 1);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_k, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
      }

      // 更新偶数号的时候计算了偶数号的resnorm，而奇数号时没有resnorm的，因此不需要额外算resnorm
      if (k > 0){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        //主要更新
        rb_kernel<<<grid_dim, block_dim, blksize*size_of_double>>>(d_u, d_b, res, 0);
        rb_kernel1<<<grid_dim, block_dim>>>(d_u, d_b, res, 1);


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_k, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);       

        //残差规约
        reduce1<<< rstep , rsize, rsize * size_of_double>>>(res, res1);
        reduce2<<< 1, rstep, rstep * size_of_double>>>(res1, d_normr); 

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_r, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
      }

      elapsed_time = time_r + time_k;
      kernel_time += elapsed_time / 1000;
      if (k>0){
        cudaMemcpy(&normr, d_normr, size_of_double, cudaMemcpyDeviceToHost);
        normr = sqrt(normr);
      }
      printf("Iteration %d, normr/normr0=%g, time cost=%g, time ite=%g, time reduce=%g\n", k, normr / normr0, elapsed_time / 1000, time_k/1000, time_r/1000);

      if (normr < RTOL * normr0 && k >= 33)
      {
        float elapsed_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        residual<<<grid_dim, block_dim, blksize*size_of_double>>>(d_u, d_b, res, 0);


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
        tsteps = k+1;
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
  cudaMemcpy(u, d_u, size_u, cudaMemcpyDeviceToHost);
  cudaFree(d_u);
  // cudaFree(d_u2);
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