# BLA on the device



## Vectors and matrices

`demo_vecs.cpp`

```cpp
Vector<double> x(n), y(n);
x = 1; y = 2;

// init from host vectors:
Vector<Dev<double>> devx(x), devy(y);

// expression templates on device
devx += 3*devy;

devx.D2H(x);
```



by now we have assignment, for VectorView


Matrix multiplication calls cublas:
```cpp
Matrix<Dev<double>> a, b, c;
c = a*b;
```
