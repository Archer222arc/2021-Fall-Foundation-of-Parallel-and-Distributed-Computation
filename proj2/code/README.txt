Implementation for Poisson equation solving
Ao Ruicheng

There are four file folders including all algorithms implemented.
Sbatch the slurm scripts with the same names of the algorithms to run. 

./openmp	all openmp algorithms
  ./results.    All numerical results
  - poisson.c     baseline
  - jacobi.c      Jacobi with merged iteration and res compute
  - jacobi2.c     Jacobi with seperate iteration and res compute
  - rb.c          Red & Black
  - linefirst.c   Line GS with line-wise Jacobi
  - linesecond.c  Line GS with plane-wise Jacobi
  - mod2.c        coordinate-wise Red & Black 
  - 8part.c       part Jacobi

----- ----- -----
./cuda

  ./cuda_jacobi  Jacobi iteration
    ./results      numerical results
    -constant.h    all constant used
    -head.h        all head file reference
    -jacobi.h      all basic functions used in main
    -utils.h       other functions used
    -reduce.cu     resnorm reduce
    -residual.cu   compute resnorm
    -initial_step  the first iterative step
    -jacobi_kernel iteration
    -test.cu	   main part, run to test

  ./cuda_rb      RB with separately stored R / B points ****benchmark!
    ./results      numerical results
    -constant.h    all constant used
    -head.h        all head file reference
    -jacobi.h      all basic functions used in main
    -utils.h       other functions used
    -reduce.cu     resnorm reduce
    -residual.cu   compute resnorm
    -initial_step  the first iterative step
    -rb_kernel.cu  rb iteration
    -initialize.cn convey value to mini grids
    -test.cu	   main part, run to test

  ./cuda_rb2      RB without separately stored R / B points
    ./results      numerical results
    -constant.h    all constant used
    -head.h        all head file reference
    -jacobi.h      all basic functions used in main
    -utils.h       other functions used
    -reduce.cu     resnorm reduce
    -residual.cu   compute resnorm
    -initial_step  the first iterative step
    -rb_kernel.cu  rb iteration
    -test.cu	   main part, run to test