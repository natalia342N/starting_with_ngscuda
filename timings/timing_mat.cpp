// ngs_nvcc -x cu -c  timing_mat.cpp ; ngs_nvcc -dlink timing_mat.o -o timing_mat_device.o -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lngcore -lngscudalib;  ngsld timing_mat.o timing_mat_device.o  -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lcudart -lcudadevrt -lcublas -lculibos -lngcore -lngscudalib

#include <fstream>
#include <ngstd.hpp>
#include <cuda_ngbla.hpp>

using namespace ngs_cuda;
using namespace ngbla;

int main()
{

  ofstream matmat("timing_matmat.txt");
  for (int n = 1024; n <= 1024*1024; n*=2)
    {
      matmat << n;
      for (int m : { 20, 50, 100, 200 })
        {
          Matrix<double> a(m,m), b(m,n), c(m,n);
          c = 0; a = 1; b = 1;
        
          Matrix<Dev<double>> deva(a), devb(b), devc(c);
        
          double est_time = 5e-6 + m*m*(n*1e-10);
          size_t runs = 0.1 / est_time + 1;
        
          Timer t("vec"+ToString(n)+ToString(m));
        
          cudaDeviceSynchronize();
          t.Start();
          for (size_t k = 0; k < runs; k++)
            MultMatMat(deva, devb, devc);
          // c = 2.0*a*b;
          cudaDeviceSynchronize();
          t.Stop();
          t.AddFlops (double(runs)*n*m*m);
        
          cout << "n = " << n << ", m = " << m << ", GFlops = " << t.GetMFlops()*1e-3 << endl;
          matmat << " " << t.GetMFlops()*1e-3  << endl;
        }
    }
}

