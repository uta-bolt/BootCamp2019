
### CPP compiler wrapper
CXX=mpicxx

### Optimization flags
FFLAGS=-O3

#######################################################################

all: bc.exec

#######################################################################

## Example 1
bc.exec: broadcast.cpp
	$(CXX) $(FFLAGS) broadcast.cpp -o broadcast.exec

clean :
	rm -f *.exec

#######################################################################
