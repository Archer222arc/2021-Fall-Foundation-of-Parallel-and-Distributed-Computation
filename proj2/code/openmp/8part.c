/*
  Foundations of Parallel and Distributed Computing, Fall 2021.
  Instructor: Prof. Chao Yang @ Peking University.
  Date: 30/11/2021
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define N 512
#define MAXITER 100
#define RTOL 1e-6
#define PI 3.14159265358979323846
#define NN N*N 
#define N2N (N+2)*(N+2)
void init_sol(double *__restrict__ b, double *__restrict__ u_exact, double *__restrict__ u)
{
    double a = N / 4.;
    double h = 1. / (N + 1);
    #pragma omp parallel for collapse(3) schedule(static,N+2)
    for (int i = 0; i < N + 2; i++)
        for (int j = 0; j < N + 2; j++)
            for (int k = 0; k < N + 2; k++)
            {
                u_exact[i * N2N + j * (N + 2) + k] = sin(a * PI * i * h) * sin(a * PI * j * h) * sin(a * PI * k * h);
                u[i * N2N + j * (N + 2) + k] = 0.;
            }
            
#pragma omp parallel for collapse(3) schedule(static,N)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                b[i * NN + j * N + k] = 3. * a * a * PI * PI * sin(a * PI * (i + 1) * h) * sin(a * PI * (j + 1) * h) * sin(a * PI * (k + 1) * h) * h * h;
            }
}

double error(double *__restrict__ u, double *__restrict__ u_exact)
{
    double tmp = 0;
#pragma omp parallel for reduction(+:tmp) schedule(static,N*N)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp += pow((u_exact[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] - u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1]), 2);
            }
    double tmp2 = 0;
#pragma omp parallel for reduction(+:tmp2) schedule(static,N*N)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp2 += pow((u_exact[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1]), 2);
            }
    return pow(tmp, 0.5) / pow(tmp2, 0.5);
}
double residual_norm(double *__restrict__ u, double *__restrict__ b)
{
    double norm2 = 0;
    double r;
#pragma omp parallel for reduction(+:norm2) private(r) collapse(3) schedule(static,N*N)
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k++)
            {
                r = b[i * NN + j * N + k] + 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1]
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    - 6.0 * u[(i + 1) * (N2N) + (j + 1) * (N + 2) + (k + 1)];
                norm2 += r * r;
            }
        }
    }
    return sqrt(norm2);
}
void gauss_seidel(double *__restrict__ u, double *__restrict__ b, double *__restrict__ bt)
{
    int x, y, z;
// 内部更新
    #pragma omp parallel for 
    for (int l = 0; l < 8; l++)
    {
        x = (l/4)*(N/2+1);    y = ((l/2)%2)*(N/2+1);    z = (l%2)*(N/2+1);
        for (int i = 0; i < N/2-1 && (i+x!=N/2) && (i+x!=N/2-1); i++)
        {   
            for (int j = 0; j < N/2-1&& (j+y!=N/2) && (j+y!=N/2-1); j++)
            {
                for (int k = 0; k < N/2 -1&& (k+z!=N/2) && (k+z!=N/2-1) ; k++)
                {
                    u[(i + x + 1) * N2N + (j + y + 1) * (N + 2) + k + z + 1] = 
                        (b[(i + x) * NN + (j + y) * N + k + z] 
                        + u[(i+x + 0) * N2N + (j+y + 1) * (N + 2) + k+z + 1] 
                        + u[(i+x + 1) * N2N + (j+y + 0) * (N + 2) + k+z + 1]
                        + u[(i+x + 1) * N2N + (j+y + 1) * (N + 2) + k+z + 0] 
                        + u[(i+x + 1) * N2N + (j+y + 1) * (N + 2) + k+z + 2]
                        + u[(i+x + 1) * N2N + (j+y + 2) * (N + 2) + k+z + 1] 
                        + u[(i+x + 2) * N2N + (j+y + 1) * (N + 2) + k+z + 1]
                        ) / 6.0;
                }
            }
        }
    }
// 中间十字更新
    for (int l = 0; l < 2; l++){
    int x = l*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = N/2-1; i < N/2+1 ; i++)
    {   
        for (int j = x; j < N/2-1+x ; j++)
        {
            for (int k = N/2-1; k < N/2+1 ; k++)
            {
                bt[(k-N/2+1)*N*N+i*N+j] = 
                    (b[i * NN + j * N + k] 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
    }
    for (int l = 0; l < 2; l++){
    int x = l*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int j = x; j < N/2-1+x ; j++)
    {
        for (int i = N/2-1; i < N/2+1 ; i++)
        {   
            for (int k = N/2-1; k < N/2+1 ; k++)
            {
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(k-N/2+1)*N*N+i*N+j];
            }
        }
    }
    }
//面 oYZ更新
    for (int l = 0; l < 2; l++){
    int x = l*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = x; i < N/2-1+x ; i++)
    {   
        for (int j = N/2-1; j < N/2+1 ; j++)
        {
            for (int k = N/2-1; k < N/2+1 ; k++)
            {
                bt[(j-N/2+1)*N*N+i*N+k] = 
                    (b[i * NN + j * N + k] 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
    }
    for (int l = 0; l < 2; l++){
    int x = l*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = x; i < N/2-1+x ; i++)
    {   
        for (int j = N/2-1; j < N/2+1 ; j++)
        {
            for (int k = N/2-1; k < N/2+1 ; k++)
            {
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(j-N/2+1)*N*N+i*N+k];
            }
        }
    }
    }
// 面 Oxy 更新
    for (int l = 0; l < 2; l++){
    int x = l*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,N/2-1)
    for (int i = N/2-1; i < N/2+1 ; i++)
    {
        for (int j = N/2-1; j < N/2+1 ; j++)
        {
            for (int k = x; k < N/2-1+x ; k++)
            {   
                bt[(i-N/2+1)*N*N+k*N+j] = 
                    (b[i * NN + j * N + k] 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
    }
    for (int l = 0; l < 2; l++){
    int x = l*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,N/2-1)
        for (int i = N/2-1; i < N/2+1 ; i++)
        {
            for (int j = N/2-1; j < N/2+1 ; j++)
            {
                for (int k = x; k < N/2-1+x ; k++)
                {   
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(i-N/2+1)*N*N+k*N+j];
            }
        }
    }
    }
//面 Oxy 更新
    for (int l = 0; l < 4; l++){
    int x = l/2*(N/2+1), y = (l%2)*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = x; i < N/2-1+x ; i++)
    {   
        for (int j = y; j < N/2-1+y ; j++)
        {
            for (int k = N/2-1; k < N/2+1 ; k++)
            {
                bt[(k-N/2+1)*N*N+i*N+j] = 
                    (b[i * NN + j * N + k] 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
    }
    for (int l = 0; l < 4; l++){
    int x = l/2*(N/2+1), y = (l%2)*(N/2+1);
    // printf("%d\n",x);
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = x; i < N/2-1+x ; i++)
    {   
        for (int j = y; j < N/2-1+y ; j++)
        {
            for (int k = N/2-1; k < N/2+1 ; k++)
            {
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(k-N/2+1)*N*N+i*N+j];
            }
        }
    }
    }
//面 oZX更新
    for (int l = 0; l < 4; l++){
    int x = l/2*(N/2+1), y = (l%2)*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,N/2-1)
    for (int i = x; i < N/2-1+x ; i++)
    {   
        for (int j = N/2-1; j < N/2+1 ; j++)
        {
            for (int k = y; k < N/2-1+y ; k++)
            {
                bt[(j-N/2+1)*N*N+i*N+k] = 
                    (b[i * NN + j * N + k] 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
    }
    for (int l = 0; l < 4; l++){
    int x = l/2*(N/2+1), y = (l%2)*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,N/2-1)
    for (int i = x; i < N/2-1+x ; i++)
    {   
        for (int j = N/2-1; j < N/2+1 ; j++)
        {
            for (int k = y; k < N/2-1+y ; k++)
            {
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(j-N/2+1)*N*N+i*N+k];
            }
        }
    }
    }
// 面 oYZ 更新
    for (int l = 0; l < 4; l++){
    int x = l/2*(N/2+1), y = (l%2)*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,N/2-1)
        for (int i = N/2-1; i < N/2+1 ; i++)
        {
            for (int j = y; j < N/2-1+y ; j++)
            {
                for (int k = x; k < N/2-1+x ; k++)
                {   
                bt[(i-N/2+1)*N*N+k*N+j] = 
                    (b[i * NN + j * N + k] 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0] 
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
    }
    for (int l = 0; l < 4; l++){
    int x = l/2*(N/2+1), y = (l%2)*(N/2+1);
#pragma omp parallel for collapse(3) schedule(static,N/2-1)
        for (int i = N/2-1; i < N/2+1 ; i++)
        {
            for (int j = y; j < N/2-1+y ; j++)
            {
                for (int k = x; k < N/2-1+x ; k++)
                {   
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(i-N/2+1)*N*N+k*N+j];
            }
        }
    }
    }


// 中间8格更新
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = N/2-1; i < N/2+1; i++)
    {
        for (int j = N/2-1; j < N/2+1; j++)
        {
            for (int k = N/2-1; k < N/2+1; k++)
            {
                bt[(i-N/2+1)*4+(j-N/2+1)*2+k-N/2+1] = 
                (b[i * NN + j * N + k] + 
                    + u[(i + 0) * N2N + (j + 1) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 0]
                    + u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * N2N + (j + 2) * (N + 2) + k + 1]
                    + u[(i + 2) * N2N + (j + 1) * (N + 2) + k + 1]
                        ) / 6.0;
            }
        }
    }
#pragma omp parallel for collapse(3) schedule(static,2)
    for (int i = N/2-1; i < N/2+1; i++)
    {
        for (int j = N/2-1; j < N/2+1; j++)
        {
            for (int k = N/2-1; k < N/2+1; k++)
            {
                u[(i + 1) * N2N + (j + 1) * (N + 2) + k + 1] = 
                bt[(i-N/2+1)*4+(j-N/2+1)*2+k-N/2+1];
            }
        }
    }
}



int main(int argc, char **argv)
{
    double * u = (double *)malloc(sizeof(double) * N2N * (N + 2));
    double * u_exact = (double *)malloc(sizeof(double) * N2N * (N + 2));
    double * b = (double *)malloc(sizeof(double) * N * N * N);
    double * bt = (double *)malloc(sizeof(double) * 2 * NN);
    omp_set_dynamic(0);
    omp_set_num_threads(8);
    init_sol(b, u_exact, u);
    double normr0 = residual_norm(u, b);
    double normr = normr0;


    int tsteps = MAXITER;
    double time0 = omp_get_wtime();
    for (int k = 0; k < MAXITER; k++)
    {
        printf("Iteration %d, normr/normr0=%g\n", k, normr / normr0);
        gauss_seidel(u, b, bt);
        normr = residual_norm(u, b);
        if (normr < RTOL * normr0 && k >= 33)
        {
            printf("Iteration %d, normr/normr0=%g\n", k + 1, normr / normr0);
            tsteps = k + 1;
            printf("Converged with %d iterations.\n", tsteps);
            break;
        }
    }
    double time1 = omp_get_wtime() - time0;

    printf("time: %g\n", time1);
    printf("Residual norm: %g\n", normr);

    long long residual_norm_bytes = sizeof(double) * (N * (N + 2) + (N*N*N)) * tsteps;
    long long gs_bytes = sizeof(double) * (N2N * (N + 2) + 2 * (N*N*N)) * tsteps;

    long long total_bytes = residual_norm_bytes + gs_bytes;
    double bandwidth = total_bytes / time1;

    printf("total bandwidth: %g GB/s\n", bandwidth / (double)(1 << 30));

    double final_normr = residual_norm(u, b); // Please ensure that this residual_norm is exact.
    printf("Final residual norm: %g\n", final_normr);
    printf("|r_n|/|r_0| = %g\n", final_normr / normr0);
    
    int num_threads = omp_get_max_threads();
    printf("openmp max num threads: %d\n", num_threads);

    free(u);
    free(u_exact);
    free(b);
    return 0;
}
