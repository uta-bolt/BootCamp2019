#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <mpi.h>

using namespace std;
int main(int argc, char **argv)
{
    int rank, size,proc,total_draws,allincirc,tag;
	double x,y,r,z,pi;
	int incirc, numincirc, num_draws,mc_draw;

	int master = 0;  /* this is the master */


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

	//num=mc_draw/size;
	numincirc=0;
	total_draws=100000;
	num_draws= total_draws/size;
	  for (int n = 0; n < num_draws; ++n) {
		  x= (double) rand() / (RAND_MAX);
		  y= (double) rand() / (RAND_MAX);
		  z=x*x + y*y;
		  if (z<1)
		  {
		    ++numincirc;
		  }
	  }

 	  MPI_Send(&numincirc, 1, MPI_INT, master, tag, MPI_COMM_WORLD); //send numincirc

	  if(rank == master) {
		 allincirc=0;
		 for (proc=0;proc< size; proc++)
		 {
		  MPI_Recv(&numincirc, 1, MPI_INT, proc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  cout << "numincirc   " << numincirc << endl ;
		  allincirc += numincirc;
		  //cout << "proc " << proc << endl ;
		 }
		 pi=4.0*((double)allincirc/total_draws);
		 cout << "Approximation of pi   " << pi;
		 cout << "Number of tasks   " << size;
 	   }

	  MPI_Finalize();
    return 0;
}
