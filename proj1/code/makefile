default: all

CC = mpicc
FLAGS = -O3 -march=native -g -Wall
OBJ = *.o
EXE = test
SRC = $(wildcard *.c)
HEAD = allreduce.h
all: ${EXE}

test: $(SRC)
	$(CC) -o $@ $^ $(FLAGS) -lm
clean:
	rm -f $(OBJ) $(EXE)

