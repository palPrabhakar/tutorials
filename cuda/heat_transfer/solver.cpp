#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>

void dump_output(double* grid, size_t xsz, size_t ysz) {
  std::ofstream fp;
  char buffer[32];
  sprintf(buffer, "dump_%s.txt", "serial");
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

void explicit_euler(double* &grid, double* &ogrid, double  grid_fo, size_t xsz, size_t ysz) {
  double nval;
  for(size_t i = 1; i < ysz-1; ++i) { // y-coordinate
    for(size_t j = 1; j < xsz-1; ++j) { // x-coordinate
      nval = grid[j+i*xsz] + grid_fo*(grid[j+1+i*xsz] + grid[j-1+i*xsz] + grid[j+(i+1)*xsz] + grid[j+(i-1)*xsz] - 4*grid[j+i*xsz]);
      ogrid[j+i*xsz] = nval;
    }
  }
  double* temp;
  temp = grid;
  grid = ogrid;
  ogrid = temp;
}

int index(int x, int y, int xsz) {
  return x + y*xsz;
}

int main() {
  size_t xsz = 21;
  size_t ysz = 21;
  int n_it = 5000;
  double alpha = 0.000019;
  double dt = 0.01;
  double dx = 0.0025;
  double dy = 0.0025;
  double grid_fo = (alpha*dt)/(dx*dy);
  int bytes = sizeof(int)*xsz*ysz;
  double left_wall = 70;
  double right_wall = 25;
  double top_wall = 0;
  double bottom_wall = 50;
  double init_temp = 10;

  double* grid = new double[bytes];
  double* ogrid = new double [bytes];

  for(size_t i = 0; i < xsz*ysz; ++i) {
    if(i < xsz) {
      grid[i] = bottom_wall;
      ogrid[i] = bottom_wall;
    }
    else if(i%xsz == 0) {
      grid[i] = left_wall;
      ogrid[i] = left_wall;
    }
    else if((i+1)%xsz == 0) {
      grid[i] = right_wall;
      ogrid[i] = right_wall;
    }
    else if (i > xsz*ysz-xsz) {
      grid[i] = top_wall;
      ogrid[i] = top_wall;
    }
    else{
      grid[i] = init_temp;
      ogrid[i] = init_temp;
    }
  }

  /* dump_output(grid, xsz, ysz); */

  for(int i = 0; i < n_it; ++i) {
    explicit_euler(grid, ogrid, grid_fo, xsz, ysz);
  }

  dump_output(grid, xsz, ysz);
 
  delete[] grid;
  delete[] ogrid;

  return 0;
}

