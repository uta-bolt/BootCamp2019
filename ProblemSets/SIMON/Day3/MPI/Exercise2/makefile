
### CPP compiler wrapper
CXX=mpicxx

### Optimization flags
FFLAGS=-O3

#######################################################################

all: allreduce.exec

#######################################################################

## Example 1
allreduce.exec: allreduce.cpp
	$(CXX) $(FFLAGS) allreduce.cpp -o allreduce.exec

clean :
	rm -f *.exec

#######################################################################
