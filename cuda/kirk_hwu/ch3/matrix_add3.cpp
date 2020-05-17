#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#define SIZE 1024

void matrix_add(float* h_A, float* h_B, float* h_C, size_t size) {
  for(size_t i = 0; i < size; ++i) {
    h_C[i] = h_A[i] + h_B[i];
  }
}

int main() {
  float* h_A;
  float* h_B;
  float* h_C;

  size_t size = SIZE*SIZE;
  size_t bytes = size*sizeof(float);

  h_A = (float *)malloc(bytes);
  h_B = (float *)malloc(bytes);
  h_C = (float *)malloc(bytes);

  for(size_t i = 0; i < size; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  matrix_add(h_A, h_B, h_C, size);

  for(size_t i = 0; i < size; ++i) {
    assert(h_C[i] == 3.0f);
    /* printf("%f ", h_C[i]); */
    /* if(i % SIZE == 0) printf("\n"); */
  }

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
