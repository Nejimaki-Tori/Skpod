#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
long int N3 = N * N * N;
double maxeps = 0.1e-7;
int itmax = 100;
int p;
double eps;

double A[N][N][N];

void relax();

void init();

void verify();

int p_thr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160};

int main(int argc, char **argv) {
    for (p = 0; p < sizeof(p_thr) / sizeof(p_thr[0]); ++p) {
        int it;
        omp_set_num_threads(p_thr[p]);
        double time_start, time_end;
        time_start = omp_get_wtime();
        omp_get_num_threads();

        init();

        for (it = 1; it <= itmax; it++) {
            eps = 0.;
            relax();
            printf("it=%4i   eps=%f\n", it, eps);
            if (eps < maxeps) break;
        }

        verify();

        time_end = omp_get_wtime();
        printf("Elapsed time: %lf.\n", time_end - time_start);

    }
    return 0;
}


void init() {
#pragma omp parallel for shared(A)
    for (int i = 0; i <= N - 1; ++i)
        for (int j = 0; j <= N - 1; ++j)
            for (int k = 0; k <= N - 1; ++k) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else
                    A[i][j][k] = (4. + i + j + k);
            }
}


void relax() {
#pragma omp parallel
    {
        int iam = omp_get_thread_num();
        int numt = omp_get_num_threads();
        for (int newi = 1; newi <= N - 2 + numt - 1; ++newi) {
            int i1 = newi - iam;
#pragma omp for reduction(max:eps)
            for (int j = 1; j <= N - 2; ++j) {
				if ((i1 > 0) && (i1 < N - 1)) { // for optimization it is here
                for (int k = 1; k <= N - 2; ++k) {                 
                        double e = A[i1][j][k]; // and not here
                        A[i1][j][k] =
                                (A[i1 - 1][j][k] + A[i1 + 1][j][k] + A[i1][j - 1][k] + A[i1][j + 1][k] +
                                 A[i1][j][k - 1] +
                                 A[i1][j][k + 1]) / 6.0;
                        eps = Max(eps, fabs(e - A[i1][j][k]));
                    }
                }
            }
        }
    }
}


void verify() {
    double s = 0.;
#pragma omp parallel for shared(A) reduction(+: s)
    for (int i = 0; i <= N - 1; ++i)
        for (int j = 0; j <= N - 1; ++j)
            for (int k = 0; k <= N - 1; ++k) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }
    printf("  S = %f\n", s);

}