#include<bits/stdc++.h>
using namespace std;
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void histogram(float *grad,float *dir, int height, int width,float *output);


