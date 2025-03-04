#include <iostream>
#include <cstdlib>
#include <vector>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  /* printf("tid: %d\n", tid); */
  if(tid < n)
    c[tid] = a[tid] + b[tid];
}

int main() {
  int n = 1 << 20;

  // Host array
  /* std::vector<int> h_a(n, 2) */
  /* std::vector<int> h_b(n, 3); */
  /* std::vector<int> h_c(n); */
  int* h_a;
  int* h_b;
  int* h_c;

  int bytes = sizeof(int)*n;

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c = (int*)malloc(bytes);

  for(int i = 0; i < n; ++i) {
    h_a[i] = 2;
    h_b[i] = 3;
  }

  //Device array
  int* d_a;
  int* d_b;
  int* d_c;

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  int block_size = 1024;
  int grid_size = (int)ceil((float)n/block_size);
  std::cout<<"grid size: "<<grid_size<<"\n";

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  std::cout<<"Mem copy successfull\n";

  vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  for(int i = 0; i < n; ++i) {
    if(h_c[i] != 5) {
      std::cout<<h_c[i]<<" "<<i<<std::endl;
      break;
    }
  }

  std::cout<<"Success!\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
