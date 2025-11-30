# Starting with ngscuda C++ programming



## Creating variables on the device

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
new/delete would be nicer?



## Vectors and matrices

`demo_vecs.cpp`

```cpp
Vector<double> x(n), y(n);
x = 1; y = 2;

// init from host vectors:
Vector<Dev<double>> devx(x), devy(y);

// expression templates on device
devx += 3*devy;
```

by now we have assignment, for VectorView


Matrix multiplication calls cublas:
```cpp
Matrix<Dev<double>> a, b, c;
c = a*b;
```