// ngs_nvcc -x cu -c  demo_vecs.cpp ; ngs_nvcc -dlink demo_vecs.o -o demo_vecs_device.o -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lngcore ;  ngsld demo_vecs.o demo_vecs_device.o  -L/opt/cuda/lib64 -lcudart -lcudadevrt -lngcore

// rm -f demo_vecs.o demo_vecs_device.o demo_vecs
// ngs_nvcc -x cu -c demo_vecs.cpp
// ngs_nvcc -dlink demo_vecs.o -o demo_vecs_device.o \
//     -L/usr/local/cuda/lib64 \
//     -lngcore
// ngsld demo_vecs.o demo_vecs_device.o \
//     -o demo_vecs \
//     -L/usr/local/cuda/lib64 \
//     -lcudart -lcudadevrt \
//     -lngcore
// ./demo_vecs


#include <cuda_ngbla.hpp>

using namespace ngs_cuda;
using namespace ngbla;


int main()
{
  Vector<double> x(10), y(10);
  for (int i = 0; i < x.Size(); i++) x(i) = 10+i;
  for (int i = 0; i < x.Size(); i++) y(i) = i;  
  
  Vector<Dev<double>> vx(x), vy(y);
    
  cout << "x = " << x << endl;
  
  vy += 4.*vx;
  vy.D2H (y);

  cout << "y = " << y << endl;
}

