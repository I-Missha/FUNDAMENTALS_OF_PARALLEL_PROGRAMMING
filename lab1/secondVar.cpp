#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cstring>
#include <iostream>

#include "mpi.h"

#define N 10000
#define EPSILON 0.00005
#define TAU (0.00001)

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

double* createVector(int elements, int vecSize) {
    double* vec = new double[vecSize];
    for (int i = 0; i < vecSize; i++) {
        vec[i] = elements;
    }
    return vec;
}

void multMatrixVec(double** matA, double* vec, double* res, int numLines, int rank, int size, int* flowsNumLines, int* startIndexes) {
    double* vecBuf = new double[flowsNumLines[0]];
    memcpy(vecBuf, vec, sizeof(double) * flowsNumLines[0]);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < numLines; j++) {
            int temp = 0;
            for (int k = 0; k < flowsNumLines[((rank + size - i) % size)]; k++) {
                temp += matA[j][startIndexes[((rank + size - i) % size)] + k] * vecBuf[k];
            }
            res[j] += temp;
        }
        
        MPI_Sendrecv_replace(vecBuf, flowsNumLines[0], MPI_DOUBLE, (rank + 1) % size, 0, (rank + size - 1) % size, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    delete[] vecBuf;

}

// 

void calcDiffVec(double* vec1, double* vec2, double* res, int vecSize) {
    for (int i  = 0; i < vecSize; i++) {
        res[i] = vec1[i] - vec2[i];
    }
}

double calcVecModule(double* vec, int vecSize, int size, int rank) {
    
    double sqSum = 0;
    for (int j = 0; j < vecSize; j++) {
        sqSum += vec[j] * vec[j];
    }
    
    double res = 0;

    MPI_Allreduce(&sqSum, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // for (int i = 0; i < size - 1; i++) {
        // MPI_Sendrecv_replace(temp, 1, MPI_DOUBLE, (rank + 1) % size, 0, (rank + size - 1) % size, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // res += *temp;
    // }
    // delete temp;
    return sqrt(res);
}

void multVecConst(double* vec, double k, int vecSize) {
    for (int i = 0; i < vecSize; i++) {
        vec[i] = vec[i] * k;
    }
}

double* findVecX(double** matA, double* vecB, int numLines, int rank, int size, int* flowsNumLines, int* startIndexes) {
    double* vecX = createVector(0, numLines);
    
    double stopFlag = 1;

    while (stopFlag > EPSILON) {
        double* temp = createVector(0, numLines);
        multMatrixVec(matA, vecX, temp, numLines, rank, size, flowsNumLines, startIndexes); 
        calcDiffVec(temp, vecB, temp, numLines); 
        stopFlag = calcVecModule(temp, numLines, size, rank) / calcVecModule(vecB, numLines, size, rank);
        multVecConst(temp, TAU, numLines);
        calcDiffVec(vecX, temp, vecX, numLines);
        delete[] temp;
    }

    return vecX;
}

void destroyVector(double* vec) {
    delete[] vec;
}

void destroyMatrixA(double** matrix, const int n) {
    for (int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void printVector(double* vec, const int n) {
    for (int i = 0; i < n; i++) {
        cout << vec[i] << endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int numLines = N / size + (rank < N % size? 1 : 0);

    int* flowsNumLines = new int[size];
    for (int i = 0; i < size; i++) {
        flowsNumLines[i] = N / size + (i < N % size? 1 : 0);
    }

    int* startIndexes = new int[size];
    startIndexes[0] = 0;
    for (int i = 1; i < size; i++) {
        startIndexes[i] = startIndexes[i - 1] + flowsNumLines[i - 1]; 
    }

    int mainDiagIndex = 0;
    for (int i = 0; i < rank; i++) {
        mainDiagIndex += flowsNumLines[i];
    }

    double **matrixA = createMatrixA(2, 1, numLines, mainDiagIndex);
    double* vecB = createVector(N + 1, numLines);
    
    // double* vec = new double[numLines];
    // for (int i = 0; i < numLines; i++) {
    //     vec[i] = 0;
    // }

    // calcDiffVec(vecB, vec, vecB, numLines);

    // multVecConst(vec, TAU, numLines);
    // multMatrixVec(matrixA, vecB, vec, numLines, rank, size, flowsNumLines, startIndexes);
    double beg, end;
    beg = MPI_Wtime();
    double* vecX = findVecX(matrixA, vecB, numLines, rank, size, flowsNumLines, startIndexes);
    end = MPI_Wtime();
    cout << end - beg << endl; 
    // printVector(vecX, numLines);
    destroyVector(vecX);
    delete[] flowsNumLines;
    delete[] startIndexes;
    destroyMatrixA(matrixA, numLines);
    // destroyVector(vecB);
    delete[] vecB;
    MPI_Finalize();
    return 0;
}

