// ngs_nvcc -x cu -c  demo_vecs.cpp ; ngs_nvcc -dlink demo_vecs.o -o demo_vecs_device.o -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lngcore ;  ngsld demo_vecs.o demo_vecs_device.o  -L/opt/cuda/lib64 -lcudart -lcudadevrt -lngcore

#include <cuda_ngbla.hpp>

using namespace ngs_cuda;
using namespace ngbla;


int main()
{
  Vector<double> x(10), y(10);
  for (int i = 0; i < x.Size(); i++) x(i) = 10+i;
  
  ngbla::Vector<Dev<double>> vx(x), vy(10);
    
  cout << "x = " << x << endl;
  
  vy.Range(0,10) = 4.*vx - vx;
  vy.D2H (y);

  cout << "y = " << y << endl;
}

