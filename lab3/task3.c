#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

// size of topology
#define NUM_DIMS 2
#define P0 4 
#define P1 4

#define N0 1000 
#define N1 1000
#define N2 1000

// mat n*m
void initMatrix(double* mat, int n, int m, double value) {
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
         mat[i * n + j] = value;
      }
   }
}

void initData(double **A, double **B, double **C, int n[3], int p[2]) {
   n[0] = N0;
   n[1] = N1;
   n[2] = N2;

   p[0] = P0;
   p[1] = P1;

   *A = (double*)malloc(N0 * N1 * sizeof(double));
   *B = (double*)malloc(N1 * N2 * sizeof(double));
   *C = (double*)malloc(N0 * N2 * sizeof(double));

   initMatrix(*A, N0, N1, 1.0);
   initMatrix(*B, N1, N2, 1.0);
   initMatrix(*C, N0, N2, 0.0);
}


void multABToC(double *A, double *B, double *C, int n[3], int p[2]) {
   double *AA, *BB, *CC;
   // AA is sub matrix of A
   int nn[2];
   // size of lines in A and B and CC
   int coords[2];
   // coords of process
   int rank;
   
   int *countc = NULL, *dispc = NULL, *countb = NULL, *dispb = NULL;
   MPI_Datatype typeb, typec, types[2];

   int blockLength[2];
   int periods[2] = {0, 0};
   int remains[2];
   // periods show us that our grid is not looped
   
   MPI_Comm comm2D, comm1D[2];
   
   MPI_Bcast(n, 3, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(p, 2, MPI_INT, 0, MPI_COMM_WORLD);
   // pass matrixes and grid demensions
   
   MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, p, periods, 0, &comm2D);
   // created new communicator comm2D
   
   MPI_Comm_rank(comm2D, &rank);
   MPI_Cart_coords(comm2D, rank, NUM_DIMS, coords);
   
   // created groups of communicators 
   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         remains[j] = i == j;
      }
      MPI_Cart_sub(comm2D, remains, &comm1D[i]);
   }
   
   // number of lines M / P0 and K / P1
   nn[0] = n[0] / p[0];
   nn[1] = n[2] / p[1];

   AA = (double*)malloc(nn[0] * n[1] * sizeof(double));
   BB = (double*)malloc(nn[1] * n[1] * sizeof(double));
   CC = (double*)malloc(nn[0] * nn[1] * sizeof(double));
   
   if (rank == 0) {

      // creating vector for b 
      MPI_Type_vector(n[1], nn[1], n[2], MPI_DOUBLE, &types[0]);
      long doubleSize;
      MPI_Type_extent(MPI_DOUBLE, &doubleSize);
      blockLength[0] = 1;
      blockLength[1] = 1;
      types[1] = MPI_UB;
      long* disp = (long*)malloc(sizeof(long) * 2);
      disp[0] = 0;
      disp[1] = doubleSize * nn[1];
      MPI_Type_struct(2, blockLength, disp, types, &typeb);
      MPI_Type_commit(&typeb);
      
      dispb = (int*)malloc(p[1] * sizeof(int));
      countb = (int*)malloc(p[1] * sizeof(int));
      for (int i = 0; i < p[1]; i++) {
         dispb[i] = i;
         countb[i] = 1;
      }
      
      // creating vector for c 
      MPI_Type_vector(nn[0], nn[1], n[2], MPI_DOUBLE, &types[0]);
      MPI_Type_struct(2, blockLength, disp, types, &typec);
      MPI_Type_commit(&typec);
      
      dispc = (int*)malloc(p[0] * p[1] * sizeof(int));
      countc = (int*)malloc(p[0] * p[1] * sizeof(int));
      for (int i = 0; i < p[0]; i++) {
         for (int j = 0; j < p[1]; j++) {
            dispc[i * p[1] + j] = (i * p[1] * nn[0] + j);
            countc[i * p[1] + j] = 1;
         }
      }

      free(disp);
   }

   if  (coords[1] == 0) {
      MPI_Scatter(A, nn[0] * n[1], MPI_DOUBLE, AA, nn[0] * n[1], MPI_DOUBLE, 0, comm1D[0]);
   }

   MPI_Barrier(MPI_COMM_WORLD);   

   if (coords[0] == 0) {
      MPI_Scatterv(B, countb, dispb, typeb, BB, n[1] * nn[1], MPI_DOUBLE, 0, comm1D[1]);
   }

   MPI_Bcast(AA, nn[0] * n[1], MPI_DOUBLE, 0, comm1D[1]);
   MPI_Bcast(BB, n[1] * nn[1], MPI_DOUBLE, 0, comm1D[0]);

   for (int i = 0; i < nn[0]; i++) {
      for (int j = 0; j < nn[1]; j++) {
         CC[i * nn[1] + j] = 0.0;
         for (int k = 0; k < n[1]; k++) {
            CC[i * nn[1] + j] = CC[i * nn[1] + j] + AA[i * n[1] + k] * BB[nn[1] * k + j];
         }
      }
   }

   MPI_Gatherv(CC, nn[0] * nn[1], MPI_DOUBLE, C, countc, dispc, typec, 0, comm2D);

   free(AA);
   free(BB);
   free(CC);

   free(dispb);
   free(countb);

   MPI_Comm_free(&comm2D);
   
   for (int i = 0; i < 2; i++) {
      MPI_Comm_free(&comm1D[i]);
   }
   
   if (rank == 0) {
      free(countc);
      free(dispc);
      MPI_Type_free(&typeb);
      MPI_Type_free(&typec);
      MPI_Type_free(&types[0]);
   }

}

int main(int argc, char** argv) {

   int size, rank, n[3], p[2];
   // n - demensions of matrix
   // p - size of computer grid
   double *A, *B, *C;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);


   int p0 = P0, p1 = P1;
   
   if (size % p0 != 0 && size %p1 != 0 && size != p0 * p1) {
      printf("wrong size of grid\n");
      return 0;
   }
   
   // initialize data on one process
   if (rank == 0) {
      initData(&A, &B, &C, n, p);
   }
   
   multABToC(A, B, C, n, p);

   if (rank == 0) {
      free(A);
      free(B);
      free(C);
   }

   MPI_Finalize();
   return 0;
}
