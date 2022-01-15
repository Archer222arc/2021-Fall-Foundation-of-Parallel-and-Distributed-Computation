#include "head.h"

// void jacobi_kernel(double* u, double* u_exact, double* b);
__global__ void rb_kernel(double* u, const double* b, double *res, int count);
__global__ void rb_kernel1(double* u, const double* b, double *res, int count);


__global__ void initial_step(double* u, const double* b);
__global__ void reduce1(const double* res, double *res0);
__global__ void reduce2(const double* res, double *res0);

__global__ void residual(const double* u, const double* b, double *res, int count);
// __global__ void residual2(double* u, const double* b, double *res);


// __host__ void test();
// void jacobi_kernel(double* u, double* u1, double* b);
// void initial_step(double* u,  double* b);
// void compute_residual(double* u, double* b, double *res);
// void reduce(double* res, int ridx, double *res0);
// void test();
// Returns the current time in microseconds
void init_sol(double *__restrict__ b, double *__restrict__ u_exact, double *__restrict__ u)
{
    double a = N / 4.;
    double h = 1. / (N + 1);
#pragma omp parallel for
    for (int i = 0; i < N + 2; i++)
        for (int j = 0; j < N + 2; j++)
            for (int k = 0; k < N + 2; k++)
            {
                u_exact[i * (N + 2) * (N + 2) + j * (N + 2) + k] = sin(a * PI * i * h) * sin(a * PI * j * h) * sin(a * PI * k * h);
                u[i * (N + 2) * (N + 2) + j * (N + 2) + k] = 0.;
            }
            
#pragma omp parallel for
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
#pragma omp parallel for reduction(+:tmp)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp += pow((u_exact[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1] - u[(i + 1) * (N + 2) * (N + 2) + (j + 1) * (N + 2) + k + 1]), 2);
            }
    double tmp2 = 0;
#pragma omp parallel for reduction(+:tmp2)
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
    double r;
#pragma omp parallel for reduction(+:norm2) private(r)
    for (int i = 0; i < N; i ++)
    {
        for (int j = 0; j < N; j ++)
        {
            for (int k = 0; k < N; k++)
            {
                r = b[i * N * N + j * N + k] + 
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

static inline long long start_timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

// Prints the time elapsed since the specified time
static inline long long stop_timer(long long start_time, char *name) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
  printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / (1000 * 1000));
  return end_time - start_time;
}