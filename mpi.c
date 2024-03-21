#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2*2*2*2*2*2*2*2 + 2)
#define TAG_PASS_FIRST 0xA
#define TAG_PASS_LAST 0xB
#define N2 (N * N)
#define N3 (N * N2)
#define NTIMES 1000


double maxeps = 0.1e-7;
int itmax = 100;
double eps;
double s = 0.;
double A[N][N][N];

void relax();
void init();
void verify();

int size, myrank, fstRow, lstRow, cntRow;

MPI_Request req_buf[4];
MPI_Status stat_buf[4];

int main(int an, char **as) {

    double time_start, time_end, tick;
    int len;
    char *name;
    name = (char*)malloc(MPI_MAX_PROCESSOR_NAME*sizeof(char));


    /* initialisation of MPI */
    MPI_Init(&an, &as);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(name, &len);
    tick = MPI_Wtick();
    time_start = MPI_Wtime();



    fstRow = (N - 2) / size * myrank + 1;
    lstRow = (N - 2) / size * (myrank + 1) + 1;
    cntRow = lstRow - fstRow;


    init();

    for (int it = 1; it <= itmax; ++it) {
        eps = 0.;
        relax();
        if (!myrank)
            printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps)
            break;
    }

    verify();

    time_end = MPI_Wtime();

    for (int i = 0; i<NTIMES; i++)
        time_end = MPI_Wtime();
    printf ("node %s, process %d: tick= %lf, time= %lf\n",
            name, myrank, tick, (time_end - time_start) / NTIMES);


    MPI_Finalize();

    return 0;
}

void init() {
    for (int i = fstRow; i < lstRow; ++i)
        for (int j = 0; j <= N - 1; ++j)
            for (int k = 0; k <= N - 1; ++k) {
                if (j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else
                    A[i][j][k] = (4. + i + j + k);
            }
}

void relax() {

    /* local variable */
    double epsTmp = 0.;

    /* process communication */
    if (myrank) {
        MPI_Irecv(A[fstRow - 1], N2, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req_buf[0]);
    }
    if (myrank != size - 1) {
        MPI_Isend(A[lstRow - 1], N2, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req_buf[2]);
    }


    if (myrank != size - 1) {
        MPI_Irecv(A[lstRow], N2, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req_buf[3]);
    }
    if (myrank) {
        MPI_Isend(A[fstRow], N2, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req_buf[1]);
    }


    int ll = 4, shift = 0;

    if (myrank == 0) {
        ll=2;
        shift=2;
    }

    if (myrank == size - 1) {
        ll -= 2;
    }

    MPI_Waitall(ll,&req_buf[shift],&stat_buf[0]);


    for (int i = fstRow; i < lstRow; ++i)
        for (int j = 1; j <= N - 2; ++j)
            for (int k = 1; k <= N - 2; ++k) {
                double e;
                e=A[i][j][k];
                A[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
                epsTmp=Max(epsTmp, fabs(e - A[i][j][k]));
            }


    //MPI_Barrier(MPI_COMM_WORLD); - no need
    //MPI_Reduce(&epsTmp, &eps, 1 , MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); //reduce all
	MPI_Allreduce(&epsTmp, &eps, 1 , MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // use this
    //MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); - no need
}

void verify() {

    /* local variable */
    double sTmp = 0.;

    for (int i = fstRow; i < lstRow; ++i)
        for (int j = 0; j <= N - 1; ++j)
            for (int k = 0; k <= N - 1; ++k) {
                sTmp = sTmp + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }
    //MPI_Barrier(MPI_COMM_WORLD); - no need
    //MPI_Reduce(&sTmp, &s, 1 , MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); - no need
	MPI_Allreduce(&sTmp, &s, 1 , MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // use this
    if (!myrank) {
        printf("  S = %f\n", s);
    }
}

