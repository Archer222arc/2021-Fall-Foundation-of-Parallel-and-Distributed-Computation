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

void init_sol(double *__restrict__ b, double *__restrict__ u_exact, double *__restrict__ u)
{
    double a = N / 4.;
    double h = 1. / (N + 1);
    #pragma omp parallel for collapse(3) schedule(static,N+2)
    for (int i = 0; i < N + 2; i++)
        for (int j = 0; j < N + 2; j++)
            for (int k = 0; k < N + 2; k++)
            {
                u_exact[i * (N + 2) * (N + 2) + j * (N + 2) + k] = sin(a * PI * i * h) * sin(a * PI * j * h) * sin(a * PI * k * h);
                u[i * (N + 2) * (N + 2) + j * (N + 2) + k] = 0.;
            }
            
#pragma omp parallel for collapse(3) schedule(static,N)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                b[i * N * N + j * N + k] = 3. * a * a * PI * PI * sin(a * PI * (i + 1) * h) * sin(a * PI * (j + 1) * h) * sin(a * PI * (k + 1) * h) * h * h;
            }
}

double error(double *__restrict__ u, double *__restrict__ u_exact)
{
    double tmp = 0;
#pragma omp parallel for reduction(+:tmp) collapse(3) schedule(static,N)
// #pragma omp parallel for reduction(+:tmp)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] - u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    double tmp2 = 0; 
#pragma omp parallel for reduction(+:tmp2) collapse(3) schedule(static,N)
// #pragma omp parallel for reduction(+:tmp2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp2 += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    return pow(tmp, 0.5) / pow(tmp2, 0.5);
}

double residual_norm(double *__restrict__ u, double *__restrict__ b)
{
    double norm2 = 0;
    #pragma omp parallel for reduction(+:norm2) collapse(3) schedule(static,N)
    // #pragma omp parallel for reduction(+:norm2) 
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            // #
            for (int k = 0; k < N; k++)
            {
                    double r = b[i * N * N + j * N + k] + 
                    + u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1]
                    + u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    - 6.0 * u[(i + 1) * ((N + 2) * (N + 2)) + (j + 1) * (N + 2) + (k + 1)];
                norm2 += r * r;
            }
        }
    }
    return sqrt(norm2);
}

void line(double *__restrict__ u, double *__restrict__ b, double *__restrict__ bt)
{

#pragma omp parallel for collapse(3) schedule(static,N)
    for (int i = 0; i < N; i ++ )
    {   
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                bt[i * N * N + j * N + k] = 
                    b[i * N * N + j * N + k]  
                +   u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 2]  
                +   u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 0];
            }
        }
    }
    for (int i = 0; i < N; i ++ )
    {   
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k ++)
            {
                u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]  = 
                    (bt[i * N * N + j * N + k] 
                    + u[(i + 0) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] 
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 0) * (N + 2) + k + 1]
                    + u[(i + 1) * (N + 2) * (N + 2) + (j + 2) * (N + 2) + k + 1] 
                    + u[(i + 2) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]
                    ) / 6.0;
            }
        }
    }
}

int main(int argc, char **argv)
{
    double * u = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
    double * u_exact = (double *)malloc(sizeof(double) * (N + 2) * (N + 2) * (N + 2));
    double * b = (double *)malloc(sizeof(double) * N * N * N);
    double * bt = (double *)malloc(sizeof(double) * N * N * N);
    init_sol(b, u_exact, u);

    double normr0 = residual_norm(u, b);
    double normr = normr0;

    int tsteps = MAXITER;
    double time0 = omp_get_wtime();
    for (int k = 0; k < MAXITER; k++)
    {
        printf("Iteration %d, normr/normr0=%g\n", k, normr / normr0);
        line(u, b, bt);
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

    long long residual_norm_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + (N * N * N)) * tsteps;
    long long gs_bytes = sizeof(double) * ((N + 2) * (N + 2) * (N + 2) + 2 * (N * N * N)) * tsteps;

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
