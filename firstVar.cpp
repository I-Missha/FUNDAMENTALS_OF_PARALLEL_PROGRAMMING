#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include "mpi.h"
#include <unistd.h> 
// #include <mpe.h> 
#define N 10
#define EPSILON 0.00005
#define TAU (0.01)

using namespace std;


double** createMatrixA(double main, double side, const int n, int mainDiagIndex) {
    double** matrixA = new double*[n];
    for (int i = 0; i < n; i++) {
        matrixA[i] = new double[N];
        for (int j = 0; j < N; j++) {
            matrixA[i][j] = j == mainDiagIndex + i? 2 : 1;
        }
    }
    return matrixA;
}

int countIndexOfFirstMatrixLine(int rank, int size) {
    int ind = 0;
    for (int i = 0; i < rank; i++) {
        ind += N / size + (i < N % size? 1 : 0);
    }
    return ind;
}

double* createVectorB(double elements, const int n) {
    double* matrix = new double[N];
    for (int i = 0; i < N; i++) {
        matrix[i] = elements;
    }
    return matrix;
}

void destroyMatrixA(double** matrix, const int n) {
    for (int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete matrix;
}

void destroyVector(double* matrix) {
    delete[] matrix;
}

void transposeMatrixA(double** matrix, const int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}

int calcIndexOfFirstMatrixLine(int rank, int size) {
    int ind = 0;
    for (int i = 0; i < rank; i++) {
        ind += N / size + (i < N % size? 1 : 0);
    }
    return ind;
}

double multVec(double* vec1, double* vec2, const int n) {
    double res = 0;
    for (int i = 0; i < N; i++) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

void multMatrixVec(double** matA, double* vecB, double* res, const int n, int startIndex) {
    for (int i = 0; i < n; i++) {
        res[i + startIndex] = multVec(matA[i], vecB, n);
    }
}

void calcDiff(double* vec1, double* vec2, double* res, const int n, int startIndex) {
    for (int i = 0; i < N; i++) {
        res[i] = vec1[i] - vec2[i];
    }
}

void printMatrix(double** mat, const int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}

void printVector(double* vec, const int n) {
    for (int i = 0; i < N; i++) {
        cout << vec[i] << endl;
    }
}


double calcVecModule(double* vec, const int n) {
    double res = 0;
    for (int i = 0; i < N; i++) {
        res += vec[i] * vec[i];
    }
    return sqrt(res);
}

void multVecConst(double* vec, double k, const int n, int startIndex) {
    for ( int i = 0; i < N; i++) {
        vec[i] *= k;
    }
}

double* findVecX(double** matA, double* vecB, const int n, int rank, int size, int* flowsN, int* startIndexes) {
    double* vecX = createVectorB(0, n);
    double* mult = createVectorB(1, n);
    double stopFlag = 1;
    while (stopFlag > EPSILON) {
        multMatrixVec(matA, vecX, mult, n, startIndexes[rank]);
        double* multCopy = new double[N];
        memcpy(multCopy, mult, sizeof(double) * N);
        MPI_Allgatherv(multCopy + startIndexes[rank], n, MPI_DOUBLE, mult, flowsN, startIndexes, MPI_DOUBLE, MPI_COMM_WORLD);
        free(multCopy);
        calcDiff(mult, vecB, mult, n, startIndexes[rank]);
        stopFlag = calcVecModule(mult, n) / calcVecModule(vecB, n);
        multVecConst(mult, TAU, n, startIndexes[rank]);
        calcDiff(vecX, mult, vecX, n, startIndexes[rank]);  
    }
    destroyVector(mult);
    return vecX;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;

    // int evtid_beginPhase1, evtid_endPhase1; int evtid_beginPhase2, evtid_endPhase2;!

    // MPE_Init_log();
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // evtid_beginPhase1 = MPE_Log_get_event_number(); !
    // evtid_endPhase1 = MPE_Log_get_event_number(); !

    // evtid_beginPhase2 = MPE_Log_get_event_number(); 
    // evtid_endPhase2 = MPE_Log_get_event_number(); 

    // MPE_Describe_state(evtid_beginPhase1, evtid_endPhase1, "Phase1", "red");!!

    // MPE_Describe_state(evtid_beginPhase2, evtid_endPhase2, "Phase2", "blue"); 

    const int n = N / size + (rank < N % size? 1 : 0);

    int* flowsN = new int[size];
    for (int i = 0; i < size; i++) {
        flowsN[i] = N / size + (i < N % size? 1 : 0);
    }

    int* startIndexes = new int[size];
    startIndexes[0] = 0;
    for (int i = 1; i < size; i++) {
        startIndexes[i] = startIndexes[i - 1] + flowsN[i - 1]; 
    }
    int mainDiagIndex = 0;
    for (int i = 0; i < rank; i++) {
        mainDiagIndex += flowsN[i];
    }

    // cout << rank << endl;

    double **matrixA = createMatrixA(2, 1, n, mainDiagIndex);
    double* vecB = createVectorB(N + 1, n);
    // MPE_Log_event(evtid_beginPhase1, rank, (char*)0);
    double* vecX = findVecX(matrixA, vecB, n, rank, size, flowsN, startIndexes);
    // MPE_Log_event(evtid_endPhase1, rank, (char*)0); 
    // MPE_Finish_log("tutor.3.clog2");  
    // printVector(vecX, n);
    destroyVector(vecX);
    free(flowsN);
    free(startIndexes);
    destroyMatrixA(matrixA, n);
    destroyVector(vecB);
    MPI_Finalize();
    return 0;
}

// firstVar.clog2
// /home/fit_opp/22202/Obryvko/task1/firstVar.clog2