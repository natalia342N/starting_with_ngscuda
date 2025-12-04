// ngs_nvcc -x cu -c  multmatmat.cpp ; ngs_nvcc -dlink multmatmat.o -o multmatmat_device.o -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lngcore -lngscudalib;  ngsld multmatmat.o multmatmat_device.o  -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lcudart -lcudadevrt -lcublas -lculibos -lngcore -lngscudalib

// rm -f multmatmat.o multmatmat_device.o multmatmat

// ngs_nvcc -x cu -c multmatmat.cpp

// ngs_nvcc -dlink multmatmat.o -o multmatmat_device.o \
//    -L/usr/local/cuda/lib64 \
//    -lngcore

// ngsld multmatmat.o multmatmat_device.o \
//    -o multmatmat \
//    -L/usr/local/cuda/lib64 \
//    -lcudart -lcudadevrt -lcublas \
//    -lngcore

// ./multmatmat 

#include <fstream>
#include <ngstd.hpp>
#include <cuda_ngbla.hpp>

using namespace ngs_cuda;
using namespace ngbla;

// 16 is faster!
#define TILE_SIZE 16 
// #define TILE_SIZE 32


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

  double sum = 0;
  
  for (int ii = 0; ii < a.Width(); ii+= TILE_SIZE)
    {
      if (row < a.Height() && ii+coli < a.Width())
        tileA[rowi][coli] = a(row,ii+coli);
      else
        tileA[rowi][coli] = 0.0;
  
      if (ii+rowi < b.Height() && col < b.Width())
        tileB[rowi][coli] = b(ii+rowi,col);
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
                              



int main()
{
  int n = 2000;
  int m = 2000;
  int k = 2000;
  
  Matrix<double> a(n,k), b(k,m), c(n,m);
  
  c = 1; a = 1; b = 1;
        
  Matrix<Dev<double>> deva(a), devb(b), devc(c);

  dim3 threads(TILE_SIZE,TILE_SIZE, 1);
  dim3 blocks(n/TILE_SIZE+1,m/TILE_SIZE+1,1);

  Timer t("matmat");
  
  cudaDeviceSynchronize();  
  t.Start();
  matmat_kernel<<<blocks,threads>>> (deva, devb, devc);
  cudaDeviceSynchronize();
  t.Stop();
  t.AddFlops (double(n)*m*k);
  
  cout << "c(0,0) = " << devc(0,0) << endl;
  cout << "c(32,0) = " << devc(n-1,m-1) << endl;
  NgProfiler::Print(stdout);
}
  
