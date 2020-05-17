#include <stdio.h>
#include "omp.h"

#define NUM_THREADS 4

int main() {
  static long num_steps = 1000000;
  double step = 1.0/(double)num_steps;
  double pi, sum[NUM_THREADS]= {0.0};
  omp_set_num_threads(NUM_THREADS);
#pragma omp parallel  
  {
    int nthreads = omp_get_num_threads();
    int twsize = num_steps/nthreads;
    int id = omp_get_thread_num();
    /* double local_sum = 0.0; */
    double x;
    for (int i = id*twsize; i < (id+1)*twsize; ++i) {
      /* printf("%d, %d\n", id, i); */
      x = (i+0.5)*step;
      sum[id] = sum[id] + 4.0/(1.0+x*x);
    }
  }
  for (int i=0; i < NUM_THREADS; ++i) {
    pi += sum[i]*step;
  }
  printf("pi: %f\n", pi);
  return 0;
}


