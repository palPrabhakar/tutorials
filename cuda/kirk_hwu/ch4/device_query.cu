#include <cstdio>
#include <iostream>

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  std::cout<<"Device count: "<<device_count<<std::endl;
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  std::cout<<"Max threads per block: "<<device_prop.maxThreadsPerBlock<<std::endl;
  std::cout<<"Multiprocessor count: "<<device_prop.multiProcessorCount<<std::endl;
  std::cout<<"Device clock rate: "<<device_prop.clockRate<<std::endl;
  return 0;
}
