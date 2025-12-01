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


    

  size_t n = 100;
  auto dev_array = Dev<int>::Malloc(n);
    
  DeviceParallelFor(n, [dev_array] DEVICE_LAMBDA (size_t tid) {
    dev_array[tid] = 2*tid;
  });

  cout << "array[5] = " << dev_array[5] << endl;

  Free (dev_array);
}
