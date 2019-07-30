#include <iostream>
#include <cstdlib>
#include <cmath>

#include <omp.h>

// function to compute the 2-norm of a vector v of length n
double norm(double *v, int n){
    double norm = 0.;

    for(int i=0; i<n; i++)
        norm += v[i]*v[i];

    return sqrt(norm);
}

// initialise v to values between -10 and 10
void initialize(double *v, int n){
    for(int i=0; i<n; i++){
        v[i] = cos(double(i)) * 10.;
    }
}


void normalize_vector(double *v, int n){
    double norm = 0.;

    // compute the norm of v
    for(int i=0; i<n; i++)
        norm += v[i]*v[i];
    norm = sqrt(norm);

    // normalize v
    for(int i=0; i<n; i++)
        v[i] /= norm;
}

void normalize_vector_omp(double *vp, int n)
{
    double normp = 0.;
    double normaux = 0.;
    #pragma omp parallel firstprivate(normaux)
    {
        #pragma omp for
        for(int i=0; i<n; i++)
        {
            normaux += vp[i]*vp[i];
        }
        #pragma omp critical
        {
        normp=normp+normaux;
        }
    }

    normp = sqrt(normp);

    // normalize v
    for(int i=0; i<n; i++){
        vp[i] /= normp;
    }

}

int main( void ){
    const int N = 10000000;
    double *v = (double*)malloc(N*sizeof(double));
    double *vp = (double*)malloc(N*sizeof(double));
    bool validated = false;

    initialize(v, N);
    double time_serial = -omp_get_wtime();
    normalize_vector(v, N); //let setial code run
    time_serial += omp_get_wtime();
    // chck the answer
    std::cout << "serial error   : " << fabs(norm(v,N) - 1.) << std::endl;
    free(v);

    int max_threads = omp_get_max_threads();
    initialize(vp, N);
    double time_parallel = -omp_get_wtime();
    normalize_vector_omp(vp, N); //let parallel code run
    time_parallel += omp_get_wtime();

    // chck the answer
    std::cout << "parallel error : " << fabs(norm(vp,N) - 1.) << std::endl;

    std::cout << max_threads     << " threads" << std::endl;
    std::cout << "serial     : " << time_serial << " seconds\t"
              << "parallel   : " << time_parallel <<  " seconds" << std::endl;
    std::cout << "speedup    : " << time_serial/time_parallel << std::endl;
    std::cout << "efficiency : " << (time_serial/time_parallel)/double(max_threads) << std::endl;


    free(vp);
    return 0;
}
