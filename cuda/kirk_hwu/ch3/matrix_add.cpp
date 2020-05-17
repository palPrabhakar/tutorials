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
  
  h_A = (float **)malloc(SIZE*sizeof(float*));
  h_B = (float **)malloc(SIZE*sizeof(float*));
  h_C = (float **)malloc(SIZE*sizeof(float*));

  for(size_t i = 0; i < SIZE; ++i) {
    h_A[i] = (float *)malloc(SIZE*sizeof(float));
    h_B[i] = (float *)malloc(SIZE*sizeof(float));
    h_C[i] = (float *)malloc(SIZE*sizeof(float));
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
    }
    printf("\n");
  }

  for(size_t i = 0; i < SIZE; ++i) {
    free(h_A[i]);
    free(h_B[i]);
    free(h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
