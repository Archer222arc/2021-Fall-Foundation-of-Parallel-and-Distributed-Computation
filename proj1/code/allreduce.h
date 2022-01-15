void recursive_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void ring_allreduce(float *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void butterfly_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void star_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void circle_allreduce(float *recvbuf, int slicenum, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
