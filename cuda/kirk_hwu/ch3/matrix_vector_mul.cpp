#include <cstdlib>
#include <cstdio>
#include <cassert>

#define SIZE 100

void matrix_vector_product(float* out_vec, float** in_mat, float* in_vec, size_t size) {
  float acc = 0.0f;
  for(size_t i = 0; i < size; ++i) {
    acc = 0.0f;
    for(size_t j = 0; j < size; ++j) {
      acc += in_mat[i][j]*in_vec[j];
      /* printf("acc: %f, mat: %f, in_vec: %f\n", acc, in_mat[i][j], in_vec[j]); */
    }
    out_vec[i] = acc;
  }
}

int main() {
  float** mat;
  float* in_vec;
  float* out_vec;
  
  mat = new float*[SIZE];
  in_vec = new float[SIZE];
  out_vec = new float[SIZE];

  for(size_t i = 0; i < SIZE; ++i) {
    mat[i] = new float[SIZE];
  }

  for(size_t i = 0; i < SIZE; ++i) {
    for(size_t j = 0; j < SIZE; ++j) {
      /* mat[i][j] = j + i*SIZE; */
      mat[i][j] = 1.0f;
    }
    in_vec[i] = static_cast<float>(i+1);
  }

  /* for(size_t i = 0; i < SIZE; ++i) { */
  /*   for(size_t j = 0; j < SIZE; ++j) { */
  /*     printf("%f ", mat[i][j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  /* for(size_t i = 0; i < SIZE; ++i) { */
  /*   printf("%f\n", in_vec[i]); */
  /* } */

  matrix_vector_product(out_vec, mat, in_vec, SIZE);

  for(size_t i = 0; i < SIZE; ++i) {
    printf("out_vec[i]: %f, i: %zu\n", out_vec[i], i);
    /* assert(out_vec[i] == i); */
  }

  for(size_t i = 0; i < SIZE; ++i) {
    delete[] mat[i];
  }

  delete[] mat;
  delete[] in_vec;
  delete[] out_vec;

  return 0;
}
