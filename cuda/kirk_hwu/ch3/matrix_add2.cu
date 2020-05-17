#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#define SIZE 32

void print_err_msg(cudaError_t err) {
  if(err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

__global__ void matrix_add3(float* d_A, float* d_B, float* d_C, size_t size, size_t h) {
  /*
     The Matrix kernel, each thread operates on 1 Matrix column
     2D Array is represented as flat 1D array 
     2D Array is processed as flat 1D array
   */
  size_t i = threadIdx.x  + blockDim.x*blockIdx.x;
  /* printf("gridDim.x: %d\n", gridDim.x); */
  for(i; i < size; i=i+h) {
    d_C[i] = d_A[i] + d_B[i];
    /* printf("i: %d, d_C[i] %f\n", i, d_C[i]); */
  }
}

__global__ void matrix_add2(float* d_A, float* d_B, float* d_C, size_t size) {
  /*
     The Matrix kernel, each thread operates on 1 Matrix row
     2D Array is represented as flat 1D array 
     2D Array is processed as flat 1D array
   */
  size_t i = threadIdx.x  + gridDim.x*blockIdx.x;
  /* printf("gridDim.x: %d\n", gridDim.x); */
  for(size_t j=i; j < i+gridDim.x && j < size; ++j) {
    d_C[j] = d_A[j] + d_B[j];
    /* printf("i: %d, d_C[i] %f\n", i, d_C[i]); */
  }
}

__global__ void matrix_add(float* d_A, float* d_B, float* d_C, size_t size) {
  /*
     The Matrix kernel, each thread operates on 1 Matrix element
     2D Array is represented as flat 1D array 
     2D Array is processed as flat 1D array
   */
  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i < size) {
    d_C[i] = d_A[i] + d_B[i];
    /* printf("i: %d, d_C[i] %f\n", i, d_C[i]); */
  }
}

int main() {
  float* h_A;
  float* h_B;
  float* h_C;

  float* d_A;
  float* d_B;
  float* d_C;

  size_t size = SIZE*SIZE;
  size_t bytes = size*sizeof(float);

  h_A = (float *)malloc(bytes);
  h_B = (float *)malloc(bytes);
  h_C = (float *)malloc(bytes);

  cudaError_t err;

  err = cudaMalloc((void **) &d_A, bytes);
  print_err_msg(err);
  err = cudaMalloc((void **) &d_B, bytes);
  print_err_msg(err);
  err = cudaMalloc((void **) &d_C, bytes);
  print_err_msg(err);

  for(size_t i = 0; i < size; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
    h_C[i] = 5.0f;
  }

  err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  print_err_msg(err);
  err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
  print_err_msg(err);

  /* int threads = SIZE; */
  /* int blocks = ceil(size/threads); */
  /* matrix_add<<<blocks, threads>>>(d_A, d_B, d_C, size); */

  /* int threads = 1; */
  /* int blocks = ceil(size/(SIZE*threads)); */
  /* matrix_add2<<<blocks, threads>>>(d_A, d_B, d_C, size); */

  int threads = 32;
  int blocks = ceil(size/(SIZE*threads));
  matrix_add2<<<blocks, threads>>>(d_A, d_B, d_C, size);

  err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
  print_err_msg(err);
  
  for(size_t i = 0; i < size; ++i) {
    assert(h_C[i] == 3.0f);
    /* printf("%f ", h_C[i]); */
    /* if(i % SIZE == 0) printf("\n"); */
  }

  free(h_A);
  free(h_B);
  free(h_C);

  err = cudaFree(d_A);
  print_err_msg(err);
  err = cudaFree(d_B);
  print_err_msg(err);
  err = cudaFree(d_C);
  print_err_msg(err);

  return 0;
}
