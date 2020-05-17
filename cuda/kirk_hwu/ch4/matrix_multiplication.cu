#include <cstdio>
#include <iostream>
#include <cmath>

#define SIZE 3
#define BLOCK_SIZE 2

__global__ void matrix_multiplication(float* d_prod, float* d_m, float* d_n, int size) {
  int col = threadIdx.x + blockDim.x*blockIdx.x; 
  int row = threadIdx.y + blockDim.y*blockIdx.y; 
  if(row < size && col < size) {
    float val = 0.0f;
    for(int k = 0; k < size; ++k) {
      val += d_m[k+row*size]*d_n[k*size+col];
    }
    d_prod[col+row*size] = val;
  }
}

int main() {
  float* h_m;
  float* h_n;
  float* h_prod;

  float* d_m;
  float* d_n;
  float* d_prod;

  size_t size = SIZE*SIZE;

  h_m = new float[size];
  h_n = new float[size];
  h_prod = new float[size];

  size_t bytes = size*sizeof(float);

  cudaMalloc((void **) &d_m, bytes);
  cudaMalloc((void **) &d_n, bytes);
  cudaMalloc((void **) &d_prod, bytes);

  for(size_t i = 0; i < size; ++i) {
    h_m[i] = i*i;
    h_n[i] = i;
  }
  
  cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, h_n, bytes, cudaMemcpyHostToDevice);

  // TODO::Kernal Call
  /* std::cout<<ceil(static_cast<float>(SIZE)/static_cast<float>(BLOCK_SIZE))<<std::endl; */
  int block_size = ceil(static_cast<float>(SIZE)/static_cast<float>(BLOCK_SIZE));
  dim3 nblocks(block_size, block_size, 1);
  dim3 nthreads(BLOCK_SIZE, BLOCK_SIZE, 1);
  matrix_multiplication<<<nblocks, nthreads>>>(d_prod, d_m, d_n, SIZE);

  cudaMemcpy(h_prod, d_prod, bytes, cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < size; ++i) {
    if(i % SIZE == 0) std::cout<<std::endl;
    std::cout<<h_prod[i]<<" ";
  }

  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_prod);

  delete[] h_m;
  delete[] h_n;
  delete[] h_prod;

  return 0;

}
