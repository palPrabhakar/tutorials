#include <cstdlib>
#include <cstdio>
#include <cassert>

#define SIZE 234

void print_err_msg(cudaError_t err, int line) {
  if(err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void matrix_vec_product(float* d_mat, float* d_in_vec, float* d_out_vec, size_t size) {
  /*
     The Matrix kernel, each thread operates on 1 Matrix row
     2D Array is represented as flat 1D array 
   */
  size_t i = threadIdx.x  + gridDim.x*blockIdx.x;
  float acc = 0.0f;
  for(size_t j = i; j < (i+gridDim.x) && j < (size*size); ++j) {
    float val = d_in_vec[j%size];
    acc += d_mat[j]*val;
  }
  d_out_vec[blockIdx.x] = acc;
}

int main() {
  float* h_mat;
  float* h_in_vec;
  float* h_out_vec;

  // device pointers
  float* d_mat;
  float* d_in_vec;
  float* d_out_vec;
  
  h_mat = new float[SIZE*SIZE];
  h_in_vec = new float[SIZE];
  h_out_vec = new float[SIZE];

  cudaError_t err;
  err = cudaMalloc((void **) &d_mat, SIZE*SIZE*sizeof(float));
  print_err_msg(err, __LINE__);
  err = cudaMalloc((void **) &d_in_vec, SIZE*sizeof(float));
  print_err_msg(err, __LINE__);
  err = cudaMalloc((void **) &d_out_vec, SIZE*sizeof(float));
  print_err_msg(err, __LINE__);
  
  for(size_t i = 0; i < SIZE; ++i) {
    for(size_t j = 0; j < SIZE; ++j) {
      /* mat[i][j] = j + i*SIZE; */
      h_mat[j+i*SIZE] = 1.0f;
    }
    h_in_vec[i] = static_cast<float>(i+1);
  }

  err = cudaMemcpy(d_mat, h_mat, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  /* print_err_msg(err); */
  err = cudaMemcpy(d_in_vec, h_in_vec, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  /* print_err_msg(err); */
  err = cudaMemcpy(d_out_vec, h_out_vec, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  /* print_err_msg(err); */

  /* for(size_t i = 0; i < SIZE; ++i) { */
  /*   for(size_t j = 0; j < SIZE; ++j) { */
  /*     printf("%f ", mat[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  /* for(size_t i = 0; i < SIZE; ++i) { */
  /*   printf("%f\n", in_vec[i]); */
  /* } */

  int threads = 1;
  int blocks = ceil((SIZE*SIZE)/(SIZE*threads));
  matrix_vec_product<<<blocks, threads>>>(d_mat, d_in_vec, d_out_vec, SIZE);

  err = cudaMemcpy(h_out_vec, d_out_vec, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  print_err_msg(err, __LINE__);

  for(size_t i = 0; i < SIZE; ++i) {
    printf("out_vec[i]: %f, i: %zu\n", h_out_vec[i], i);
    /* assert(out_vec[i] == i); */
  }

  delete[] h_mat;
  delete[] h_in_vec;
  delete[] h_out_vec;

  err = cudaFree(d_mat);
  /* print_err_msg(err); */
  err = cudaFree(d_in_vec);
  /* print_err_msg(err); */
  err = cudaFree(d_out_vec);
  /* print_err_msg(err); */

  return 0;
}
