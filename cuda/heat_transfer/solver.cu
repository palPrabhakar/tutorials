#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <helper_cuda.h>

#define X_SIZE 21
#define Y_SIZE 21

#define N_THREADS 128

__global__ void explicit_euler(double* d_grid, double* d_tgrid, double grid_fo, int xsize, int ysize) {
  /* printf("d_grid: %p, d_tgrid: %p, gridFo: %f, xsize: %d, ysize: %d\n", &grid_fo, &d_tgrid, grid_fo, xsize, ysize); */ 
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  
  /* if(idx < xsize*ysize) { */
  /*   printf("idx: %d, value: %f\n", idx, d_grid[idx]); */
  /* } */

  if((idx < xsize*ysize) && (idx-xsize > 0) && (idx+xsize < xsize*ysize) && ((idx+1)%xsize != 0) && (idx%xsize != 0)) {
    d_tgrid[idx] = d_grid[idx] + grid_fo*(d_grid[idx+1] + d_grid[idx-1] + d_grid[idx+xsize] + d_grid[idx-xsize] - 4*d_grid[idx]);
  }
}

void dump_output(double* grid, size_t xsz, size_t ysz) {
  std::ofstream fp;
  char buffer[32];
  sprintf(buffer, "dump_%s.txt", "cuda");
  fp.open(buffer);
  if(fp.is_open()) {
    for(size_t i = 0; i  < ysz; ++i) {
      for(size_t j = 0; j < xsz; ++j) {
        fp<<std::setprecision(10)<<grid[j+i*xsz]<<" ";
      }
      fp<<"\n";
    }
    fp.close();
  } 
}


int main() {
  const size_t xsize = X_SIZE;
  const size_t ysize = Y_SIZE;
  constexpr size_t size = xsize*ysize;
  const int n_it = 5000;
  const double alpha = 0.000019;
  const double dt = 0.01;
  const double dx = 0.0025;
  const double dy = 0.0025;
  const double grid_fo = (alpha*dt)/(dx*dy);
  const double left_wall = 70;
  const double right_wall = 25;
  const double top_wall = 0;
  const double bottom_wall = 50;
  const double init_temp = 10;

  double* h_grid;
  double* h_tgrid;

  double* d_grid;
  double* d_tgrid;

  const size_t bytes = sizeof(double)*size;

  h_grid = (double *)malloc(bytes);
  h_tgrid = (double *)malloc(bytes);

  checkCudaErrors(cudaMalloc((void **) &d_grid, bytes));
  checkCudaErrors(cudaMalloc((void **) &d_tgrid, bytes));

  for(size_t i = 0; i < size; ++i) {
    if(i < xsize) {
      h_grid[i] = bottom_wall;
      h_tgrid[i] = bottom_wall;
    }
    else if(i%xsize == 0) {
      h_grid[i] = left_wall;
      h_tgrid[i] = left_wall;
    }
    else if((i+1)%xsize == 0) {
      h_grid[i] = right_wall;
      h_tgrid[i] = right_wall;
    }
    else if (i > ((xsize*ysize)-xsize)) {
      h_grid[i] = top_wall;
      h_tgrid[i] = top_wall;
    }
    else{
      h_grid[i] = init_temp;
      h_tgrid[i] = init_temp;
    }
  }

  checkCudaErrors(cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_tgrid, h_tgrid, bytes, cudaMemcpyHostToDevice));

  int nthreads = N_THREADS;
  int nblocks = ceil(static_cast<double>(size)/static_cast<double>(nthreads));
  /* printf("nblocks: %d, nthreads: %d, dval: %f\n", nblocks, nthreads, h_grid[415+xsize]); */
  double* temp;
  for(int i = 0; i < n_it; ++i) {
    explicit_euler<<<nblocks, nthreads>>>(d_grid, d_tgrid, grid_fo, xsize, ysize);
    temp = d_grid;
    d_grid = d_tgrid;
    d_tgrid = temp;
  }

  checkCudaErrors(cudaMemcpy(h_tgrid, d_tgrid, bytes, cudaMemcpyDeviceToHost));

  dump_output(h_tgrid, xsize, ysize);

  cudaFree(d_grid);
  cudaFree(d_tgrid);

  free(h_grid);
  free(h_tgrid);

  return 0;
}

