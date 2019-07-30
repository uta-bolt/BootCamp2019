#include <stdio.h>
#include <mpi.h>
#include <iostream>
using namespace std;
int main(int argc, char *argv[])
{
    int rank;
    double data;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int root_process=0;
    data=40;
    /* broadcast the value of data of rank 0 to all ranks */
    MPI_Bcast(&data,1,MPI_INT,root_process,MPI_COMM_WORLD);
    cout << "I am rank" << rank << "and the value is  " << data << endl;
    MPI_Finalize();
    return 0;
}
