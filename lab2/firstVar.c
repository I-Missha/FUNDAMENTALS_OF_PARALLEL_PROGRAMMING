#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <bits/time.h>
#define N 100
#define TAU 0.001
#define EPSILON 0.000001

double* createMatrix() {
  double* mat = (double*)malloc(N * N * sizeof(double));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) { 
      mat[i * N + j] = i == j ? 2 : 1;
    } 
  }  
  return mat;
}

double* createVector(double el) {
  double* vec = (double*)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    vec[i] = el;
  }
  return vec;
}

void multMatrixVec(double* mat, double* vec, double* res) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    double temp = 0;
    for (int j = 0; j < N; j++) {
      temp += mat[i * N + j] * vec[j];
    } 
    res[i] = temp;
  }

}


double calcNorm(double* vec) {
  double norm = 0;
  #pragma omp parallel for reduction(+:norm)
  for (int i = 0; i < N; i++) {
    norm += vec[i] * vec[i];
  }  

  return sqrt(norm); 
} 

void calcDiff(double* vec1, double* vec2) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    vec1[i] -= vec2[i]; 
  } 
}

void multVecConst(double* vec, double multiplier) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    vec[i] = vec[i] * multiplier; 
  }
}

double* findVec(double* mat, double* vecB) {
  double* vecX = createVector(0.0);
  double normX = 1.0; 
  double normB = calcNorm(vecB);
  double* temp = createVector(0.0); 
  while(normX / normB > EPSILON) { 
    multMatrixVec(mat, vecX, temp); 
    calcDiff(temp, vecB);
    normX = calcNorm(temp);
    multVecConst(temp, TAU);
    calcDiff(vecX, temp);
  }
  free(temp);  
  return vecX;
}

void printVec(double* vec) {
  for (int i = 0; i < N; i++) {
    printf("%f\n", vec[i]);
  }
}

void printMat(double* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", mat[i * N + j]);
    } 
    printf("\n");
  }
}

int main() {
  double* matA = createMatrix();
  double* vecB = createVector((double)(N + 1));
  struct timespec beg, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &beg);

  double* vecX = findVec(matA, vecB);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double time = (double)(end.tv_sec - beg.tv_sec) + (double)(end.tv_nsec - beg.tv_nsec) / 1000000000; 
  printf("%lf \n", time);
  free(matA);
  free(vecB);
  free(vecX);
  return 0;
}
