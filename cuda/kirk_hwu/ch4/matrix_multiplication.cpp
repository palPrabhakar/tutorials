#include <iostream>
#include <cstdlib>

#define SIZE 3

void matrix_multiplication(float* mat_prod, float* mat_m, float* mat_n, size_t size) {
  for(size_t i = 0; i < size; ++i) {
    for(size_t j = 0; j < size; ++j) {
      double val = 0.0;
      for(size_t k = 0; k < size; ++k) {
        val += mat_m[k+i*size]*mat_n[k*size+j];
      }
      mat_prod[j+i*size] = val;
    }
  }
}

int main() {
  float* mat_m;
  float* mat_n;
  float* mat_prod;

  size_t size = SIZE*SIZE;

  mat_m = new float[size];
  mat_n = new float[size];
  mat_prod = new float[size];

  for(size_t i = 0; i < size; ++i) {
    mat_m[i] = i*i;
    mat_n[i] = i;
  }

  matrix_multiplication(mat_prod, mat_m, mat_n, SIZE);

  for(size_t i = 0; i < size; ++i) {
    if(i % SIZE == 0) std::cout<<std::endl;
    std::cout<<mat_prod[i]<<" ";
  }

  delete[] mat_m;
  delete[] mat_n;
  delete[] mat_prod;

  return 0;
  
}
