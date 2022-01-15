Implementation for allreduce with only send/recv
Ao Ruicheng

The allreduce.c all has included test code, to directly call them, just delete the comment and sbatch the slurm scripts with the same name!
allreduce_butterfly.c -- butterfly & with batch test
allreduce_circle.c    -- circle	& with batch test
allreduce_ring.c      -- ring & with batch test
allreduce_star.c.     -- star & with batch test
allreduce_recursive.c -- recursive

test.c                -- batch test with out-place/ in-place
 
./results includes the results in txt

To run batch test, you needs sbatch allreduce1(2,3,4).sh
To separately test algorithms, open .c file and delete the comments, then sbatch the slurm file with same name

WARNIG: Dont run batch test while there are uncomment "main" in any allreduce.c !!!!!