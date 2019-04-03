#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

__global__ void convolution( float *image, int height, int width, float *output	);