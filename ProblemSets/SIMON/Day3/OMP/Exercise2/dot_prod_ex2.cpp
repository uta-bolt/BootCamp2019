#include <iostream>
#include <vector>

#include <omp.h>

int main(void){
    const int N = 10000;
    std::vector<double> a(N);
    std::vector<double> b(N);

    int num_threads = omp_get_max_threads();
    std::cout << "dot of vectors with length " << N  << " with " << num_threads << " threads" << std::endl;

    // initialize the vectors
    for(int i=0; i<N; i++) {
        a[i] = 1./2.;
        b[i] = double(i+1);
    }

    double time = -omp_get_wtime();
    double dotaux=0.;
    double dot=0.;
    #pragma omp parallel firstprivate(dotaux)
    {
        #pragma omp for
            for(int i=0; i<N; i++)
            {
                dotaux += a[i] * b[i];
            }
        #pragma omp critical
            {
            dot=dot+dotaux;    
            }

    }
    time += omp_get_wtime();

    // use formula for sum of arithmetic sequence: sum(1:n) = (n+1)*n/2
    double expected = double(N+1)*double(N)/4.;
    std::cout << "dot product " << dot
              << (dot==expected ? " which matches the expected value "
                                : " which does not match the expected value ")
              << expected << std::endl;
    std::cout << "that took " << time << " seconds" << std::endl;
    return 0;
}
