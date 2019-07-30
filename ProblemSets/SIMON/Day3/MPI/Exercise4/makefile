### CPP compiler wrapper
CXX=mpicxx -std=c++11

### Optimization flags
FFLAGS=-O3

#######################################################################

all: mc_mpi.exec

#######################################################################

## Example 1
mc_mpi.exec: mc_mpi.cpp
	$(CXX) $(FFLAGS) mc_mpi.cpp -o mc_mpi.exec

clean :
	rm -f *.exec
#######################################################################
