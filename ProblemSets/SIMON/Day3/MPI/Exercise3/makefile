### CPP compiler wrapper
CXX=mpicxx

### Optimization flags
FFLAGS=-O3

#######################################################################

all: scatter.exec

#######################################################################

## Example 1
scatter.exec: scatter.cpp
	$(CXX) $(FFLAGS) scatter.cpp -o scatter.exec

clean :
	rm -f *.exec

#######################################################################
