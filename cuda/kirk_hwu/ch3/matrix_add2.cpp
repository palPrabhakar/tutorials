#include <cstdlib>
#include <cstdio>
#include <cassert>

#define SIZE 10

void matrix_add(float** h_A, float** h_B, float** h_C) {
  for(size_t i = 0; i < SIZE; ++i) {
    for(size_t j = 0; j < SIZE; ++j) {
      h_C[i][j] = h_A[i][j] + h_B[i][j];
    }
  }
}

int main() {
  float** h_A;
  float** h_B;
  float** h_C;
  
  h_A = new float*[SIZE];
  h_B = new float*[SIZE];
  h_C = new float*[SIZE];

  for(size_t i = 0; i < SIZE; ++i) {
    h_A[i] = new float[SIZE];
    h_B[i] = new float[SIZE];
    h_C[i] = new float[SIZE];
  }

  for(size_t i = 0; i < SIZE; ++i) {
    for(size_t j = 0; j < SIZE; ++j) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 2.0f;
    }
  }
  
  matrix_add(h_A, h_B, h_C);

  for(size_t i = 0; i < SIZE; ++i) {
    for(size_t j = 0; j < SIZE; ++j) {
      assert(h_C[i][j] == 3.0f);
      /* printf("%f\t", h_C[i][j]); */
    }
    /* printf("\n"); */
  }

  for(size_t i = 0; i < SIZE; ++i) {
    delete[] h_A[i];
    delete[] h_B[i];
    delete[] h_C[i];
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
