# Creating variables on the device

from `demo_devT.cpp`
```cpp
// allocate on device memory
Dev<double> * dx = Dev<double>::Malloc(2);

// transfer with explicit H2D / D2H calls:
dx->H2D(3.8);
cout << "dx = " << dx->D2H() << endl;
  
// or with assignment and conversion:
dx[1] = 4.2;
cout << "dx = " << double(dx[1]) << endl;
  
Free(dx);
```

To avoid  accidentally creating an `Dev<T>` object on the host, it has deleted default constructor.
This disallows also creation via new/delete.


## A parallel loop on the device

```cpp
size_t n = 100;
auto dev_array = Dev<int>::Malloc(n);
    
DeviceParallelFor(n, [dev_array] DEVICE_LAMBDA (size_t tid) {
    dev_array[tid] = 2*tid;
});

cout << "array[5] = " << dev_array[5] << endl;

Free (dev_array);
```


