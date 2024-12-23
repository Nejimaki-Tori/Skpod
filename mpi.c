#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (66)
#define N2 (N * N)
#define N3 (N * N2)

double maxeps = 0.1e-7;
int itmax = 100;
double eps;
double s = 0., e;
double A[N][N][N];

void relax();
void init();
void verify();
int ranksize, myrank;
int it;

MPI_Request req_buf[6];
MPI_Status stat_buf[6];

MPI_Comm main_comm;

int main(int argc, char **argv) {
    /* initialisation of MPI */
    MPI_Init(&argc, &argv);
	main_comm = MPI_COMM_WORLD;
	
    MPI_Comm_size(main_comm, &ranksize);
	MPI_Comm_rank(main_comm, &myrank);

	if (myrank == 0){
		init();
	}
	MPI_Bcast(A, N3, MPI_DOUBLE, 0, main_comm);

    for (it = 0; it < itmax; ++it) {
		
        eps = 0.;
		
        relax();

        if (!myrank)
            printf("it=%4i   eps=%f\n", it, eps);
		
        if (eps < maxeps)
            break;
    }
	
    verify();

    MPI_Finalize();

    return 0;
}

    void init()
    {

        for(int k=0; k<=N-1; k++)
        for(int j=0; j<=N-1; j++)
        for(int i=0; i<=N-1; i++)
        {
            if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
            A[i][j][k]= 0.;
            else A[i][j][k]= ( 4. + i + j + k) ;
        }
    }

void relax() {
    double localEps = 0.;

	for (int i = 1; i < N - 1; ++i) {
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) break;
			for (int k = 1; k < N - 1; ++k) {
				
				if (myrank > 0) {
                    MPI_Recv(&A[i][j - 1][k], 1, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[0]);
                }
				
				if (!myrank && j != 1 && ranksize != 1) {
					MPI_Recv(&A[i][j - 1][k], 1, MPI_DOUBLE, ranksize - 1, 0, main_comm, &stat_buf[1]);
					
				}
				
				double oldVal = A[i][j][k];
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + 
							  A[i][j - 1][k] + A[i][j + 1][k] +
                              A[i][j][k - 1] + A[i][j][k + 1]) / 6.0;
                localEps = Max(localEps, fabs(oldVal - A[i][j][k]));
				
				if (myrank < ranksize - 1 && (((N - 2) % ranksize == 0) || j != N - 2)) {
                    MPI_Send(&A[i][j][k], 1, MPI_DOUBLE, myrank + 1, 0, main_comm);
                }
				if (myrank == ranksize - 1 && myrank && j != N - 2) {
                    MPI_Send(&A[i][j][k], 1, MPI_DOUBLE, 0, 0, main_comm);
                }

			}
		}
	for (int j = myrank + 1; j < N - 1; j += ranksize) {
		if (j >= N - 1) { 
			break;
		}
		if (!myrank && ranksize > 1 && (((N - 2) % ranksize == 0) || j != N - 2)){
			MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 1, 0, main_comm);
		}
		if (!myrank && j != 1 && ranksize > 2) {
			MPI_Send(&A[i][j][0], N, MPI_DOUBLE, ranksize - 1, 0, main_comm);
		}
	}
	
	if ((myrank == ranksize - 1 || myrank == 1) && ranksize > 1) {
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) { 
				break;
			}
			
			if (myrank == 1) {
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, 0, 0, main_comm, &stat_buf[1]);
				
			} else if (j != N - 2 && ranksize > 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, 0, 0, main_comm, &stat_buf[0]);
			}
			
		}
		
	}
	if (myrank > 0) {
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			
			if (j >= N - 1) { 
				break;
			}
			if (myrank == 1 && ranksize > 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 2, 0, main_comm);
				
			} else if (myrank != ranksize - 1 && ranksize > 2) {
				if ((N - 2) % ranksize == 0 || j != N - 2){
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[1]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm);
				} else if (j == N - 2) {
					MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[1]);
				}
			} else if (ranksize > 2) {
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[0]);
			}
			
		}
	}
	
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) { 
				break;
			}
			if (myrank == ranksize - 1 && ranksize > 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			} else if (myrank != 1 && myrank != 0 && ranksize > 2) {	
			if ((N - 2) % ranksize == 0 || j != N - 2){
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm, &stat_buf[1]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			} else if (j == N - 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			}
			} else if (myrank == 1 && ranksize > 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm, &stat_buf[0]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);		
			} else if (myrank == 1) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);
				
			} else if (!myrank && ranksize > 1 && j != N - 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, 1, 0, main_comm, &stat_buf[0]);
				
			}
		}
		
		if (myrank == ranksize - 1 && ranksize > 2) {
			for (int j = myrank + 1; j < N - 1; j += ranksize) {
				if (j >= N - 1) { 
					break;
				}
				if (j != N - 2) {
					MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);
					
				}
			}
		}
		
		if (myrank == 0 && ranksize > 2) {
			for (int j = myrank + 1; j < N - 1; j += ranksize) {
				if (j >= N - 1) { 
					break;
				}
				if (j != 1) {
					MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, ranksize - 1, 0, main_comm, &stat_buf[0]);
				}
			}
		}
	}
	
    MPI_Allreduce(&localEps, &eps, 1 , MPI_DOUBLE, MPI_MAX, main_comm);
}

void verify() {

    double sTmp = 0.;
	if (myrank != ranksize){
	for(int i = 0; i < N - 1; i++) 
		for (int j = myrank + 1; j < N - 1; j += ranksize){
			if (j >= N - 1)
				break;
			for(int k = 0; k < N - 1; k++){
                sTmp = sTmp + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }
		}
	}
    MPI_Allreduce(&sTmp, &s, 1 , MPI_DOUBLE, MPI_SUM, main_comm);
	if (!myrank) {
        printf("  S = %f\n", s);
    }
}
