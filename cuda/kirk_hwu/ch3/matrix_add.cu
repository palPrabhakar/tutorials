#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#define SIZE 32

__global__ void matrix_add(float** d_A, float** d_B, float** d_C, size_t size) {
  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  printf("i: %d, j: %d, d_A[i][j]: %f\n", i, j, d_A[i][j]);
  /* if(i < size && j < size) */
  /*   d_C[i][j] = d_A[i][j] + d_B[i][j]; */
}

void print_err_msg(cudaError_t err) {
  if(err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

int main() {
  float** h_A;
  float** h_B;
  float** h_C;

  float** d_A;
  float** d_B;
  float** d_C;

  float** s_A;
  float** s_B;
  float** s_C;
  
  h_A = new float*[SIZE];
  h_B = new float*[SIZE];
  h_C = new float*[SIZE];

  s_A = (float **)malloc(SIZE*sizeof(float*));
  s_B = (float **)malloc(SIZE*sizeof(float*));
  s_C = (float **)malloc(SIZE*sizeof(float*));

  cudaError_t err;

  err = cudaMalloc((void**) &d_A, SIZE*sizeof(float*));
  print_err_msg(err);
  err = cudaMalloc((void**) &d_B, SIZE*sizeof(float*));
  print_err_msg(err);
  err = cudaMalloc((void**) &d_C, SIZE*sizeof(float*));
  print_err_msg(err);

  err = cudaMemcpy(s_A, d_A, SIZE*sizeof(float*), cudaMemcpyDeviceToHost);
  err = cudaMemcpy(s_B, d_B, SIZE*sizeof(float*), cudaMemcpyDeviceToHost);
  err = cudaMemcpy(s_C, d_C, SIZE*sizeof(float*), cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < SIZE; ++i) {
    h_A[i] = new float[SIZE];
    h_B[i] = new float[SIZE];
    h_C[i] = new float[SIZE];
    err = cudaMalloc((void **) &s_A[i], SIZE*sizeof(float));
    print_err_msg(err);
    err = cudaMalloc((void **) &s_B[i], SIZE*sizeof(float));
    print_err_msg(err);
    err = cudaMalloc((void **) &s_C[i], SIZE*sizeof(float));
    print_err_msg(err);
  }

  for(size_t i = 0; i < SIZE; ++i) {
    for(size_t j = 0; j < SIZE; ++j) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 2.0f;
    }
  }

  for(size_t i = 0; i < SIZE; ++i) {
    cudaMemcpy(s_A[i], h_A[i], SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s_B[i], h_B[i], SIZE*sizeof(float), cudaMemcpyHostToDevice);
  }

  
  int threads = 32;
  dim3 nthreads(threads, threads);
  int blocks = ceil(SIZE/threads);
  dim3 nblocks(blocks, blocks);
  matrix_add<<<nblocks, nthreads>>>(d_A, d_B, d_C, SIZE);

  for(size_t i = 0; i < SIZE; ++i) {
    cudaMemcpy(h_C[i], s_C[i], SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  }

  /* for(size_t i = 0; i < SIZE; ++i) { */
  /*   for(size_t j = 0; j < SIZE; ++j) { */
  /*     /1* assert(h_C[i][j] == 3.0f); *1/ */
  /*     printf("%f\t", h_C[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  for(size_t i = 0; i < SIZE; ++i) {
    delete[] h_A[i];
    delete[] h_B[i];
    delete[] h_C[i];
    err = cudaFree(s_A[i]);
    print_err_msg(err);
    err = cudaFree(s_B[i]);
    print_err_msg(err);
    err = cudaFree(s_C[i]);
    print_err_msg(err);
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  err = cudaFree(d_A);
  print_err_msg(err);
  err = cudaFree(d_B);
  print_err_msg(err);
  err = cudaFree(d_C);
  print_err_msg(err);

  return 0;
}
