# Starting with ngscuda C++ programming



[Documentation](https://ngsolve.github.io/starting_with_ngscuda/intro.html)



Some simple use-cases:



```cpp
size_t n = 100;
auto dev_array = Dev<int>::Malloc(n);
    
DeviceParallelFor(n, [dev_array] DEVICE_LAMBDA (size_t tid) {
    dev_array[tid] = 2*tid;
});

cout << "array[5] = " << dev_array[5] << endl;

Free (dev_array);
```

