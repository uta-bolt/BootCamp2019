#include <stdio.h>
#include <mpi.h>
#include <iostream>
using namespace std;

int main (int argc, char *argv[])
{
    int myrank,  input, size,result;
    int sum;


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute sum of all ranks. */
    input=myrank+1;
    MPI_Reduce(&input,&result,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    if (myrank==0){
        cout << "Rank  " << myrank << "  Sum=  "<< result << endl;
    }
    MPI_Finalize();
    return 0;
}
