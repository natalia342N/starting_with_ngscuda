// ngs_nvcc -x cu -c  demo_devT.cpp ; ngs_nvcc -dlink demo_devT.o -o demo_devT_device.o -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lngcore ;  ngsld demo_devT.o demo_devT_device.o  -L/opt/cuda/lib64 -lcudart -lcudadevrt -lngcore

#include <cuda_ngbla.hpp>

using namespace ngs_cuda;
using namespace ngbla;


int main()
{
  // allocate on device memory
  Dev<double> * dx = Dev<double>::Malloc(2);

  // transfer with explicit H2D / D2H calls:
  dx->H2D(3.8);
  cout << "dx = " << dx->D2H() << endl;
  
  // with assignment and conversion:
  dx[1] = 4.2;
  cout << "dx = " << double(dx[1]) << endl;
  
  Free(dx);
}
