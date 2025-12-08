// ngs_nvcc -x cu -c multmatmat.cpp
// ngs_nvcc -dlink multmatmat.o -o multmatmat_device.o \
//    -L/usr/local/cuda/lib64 \
//    -lngcore
// ngsld multmatmat.o multmatmat_device.o \
//    -o multmatmat \
//    -L/usr/local/cuda/lib64 \
//    -lcudart -lcudadevrt -lcublas \
//    -lngcore

#include <fstream>
#include <vector>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include <ngstd.hpp>
#include <cuda_ngbla.hpp>

using namespace std;
using namespace ngs_cuda;
using namespace ngbla;

// 16 is faster!
#define TILE_SIZE 16

__global__ void matmat_kernel(SliceMatrix<Dev<double>> a,
                              SliceMatrix<Dev<double>> b,
                              SliceMatrix<Dev<double>> c)
{
  int colblock = blockIdx.x;
  int rowblock = blockIdx.y;

  int coli = threadIdx.x;
  int rowi = threadIdx.y;

  int col = colblock*TILE_SIZE + coli;
  int row = rowblock*TILE_SIZE + rowi;

  __shared__ double tileA[TILE_SIZE][TILE_SIZE];
  __shared__ double tileB[TILE_SIZE][TILE_SIZE];

  double sum = 0.0;

  for (int ii = 0; ii < a.Width(); ii += TILE_SIZE)
  {
    if (row < a.Height() && ii+coli < a.Width())
      tileA[rowi][coli] = a(row, ii+coli);
    else
      tileA[rowi][coli] = 0.0;

    if (ii+rowi < b.Height() && col < b.Width())
      tileB[rowi][coli] = b(ii+rowi, col);
    else
      tileB[rowi][coli] = 0.0;

    __syncthreads();

    if (row < c.Height() && col < c.Width())
    {
      for (int i = 0; i < TILE_SIZE; i++)
        sum += tileA[rowi][i] * tileB[i][coli];
    }

    __syncthreads();
  }

  if (row < c.Height() && col < c.Width())
    c(row,col) = sum;
}

void run_single_original()
{
  int n = 2000;
  int m = 2000;
  int k = 2000;

  Matrix<double> a(n,k), b(k,m), c(n,m);

  c = 1.0; a = 1.0; b = 1.0;

  Matrix<Dev<double>> deva(a), devb(b), devc(c);

  dim3 threads(TILE_SIZE, TILE_SIZE, 1);
  dim3 blocks(n/TILE_SIZE+1, m/TILE_SIZE+1, 1);

  Timer t("matmat");

  cudaDeviceSynchronize();
  t.Start();
  matmat_kernel<<<blocks,threads>>>(deva, devb, devc);
  cudaDeviceSynchronize();
  t.Stop();
  t.AddFlops(double(n)*m*k);

  cout << "Single run (original):" << endl;
  cout << "  c(0,0)     = " << devc(0,0) << endl;
  cout << "  c(n-1,m-1) = " << devc(n-1,m-1) << endl;
  NgProfiler::Print(stdout);
}

void run_benchmark(int m)
{
  // n-values (matrix width, x-axis)
  vector<int> Ns = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
  int repeats = 10;   // repetitions to average over

  string fname = "timings_matmat_m" + to_string(m) + ".txt";
  ofstream ofs(fname);
  ofs << "# m = " << m << "\n";
  ofs << "# n   avg_time_sec   GFLOPs\n";

  for (int n : Ns)
  {
    cout << "Benchmark: m = " << m << ", n = " << n << " ..." << endl;

    // A: m×m, B: m×n, C: m×n
    Matrix<double> a(m, m), b(m, n), c(m, n);
    a = 1.0; b = 1.0; c = 0.0;

    Matrix<Dev<double>> deva(a), devb(b), devc(c);

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE,
                (m + TILE_SIZE - 1) / TILE_SIZE,
                1);

    Timer t("matmat_bench");

    matmat_kernel<<<blocks, threads>>>(deva, devb, devc);
    cudaDeviceSynchronize();

    double ops = double(m) * double(m) * double(n) * double(repeats);

    cudaDeviceSynchronize();
    double t0 = WallTime();
    t.Start();

    for (int r = 0; r < repeats; r++)
      matmat_kernel<<<blocks, threads>>>(deva, devb, devc);

    cudaDeviceSynchronize();
    t.Stop();
    double t1 = WallTime();

    t.AddFlops(ops);  
    double total_sec = t1 - t0;
    double avg_sec   = total_sec / repeats;

    double flops  = 2.0 * ops;
    double gflops = flops / total_sec / 1e9;

    cout << "  avg time = " << avg_sec*1e3 << " ms,  "
         << "perf = " << gflops << " GFLOP/s" << endl;

    ofs << n << "  " << avg_sec << "  " << gflops << "\n";
  }

  ofs.close();
  cout << "Benchmark data written to " << fname << endl;
}


int main(int argc, char** argv)
{
  run_single_original();

  if (argc > 1)
  {
    int m = atoi(argv[1]);     // e.g. ./multmatmat 100
    if (m > 0)
      run_benchmark(m);
    else
      cerr << "Invalid m given, skipping benchmark." << endl;
  }

  return 0;
}
