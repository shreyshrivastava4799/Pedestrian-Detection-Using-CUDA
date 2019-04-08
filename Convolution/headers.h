#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

__global__ void convolution( float *image, int height, int width, int blockX,
							 int blockY, float *outputMag, float *outputAng,
							 int imgRows, int imgCols);

__global__ void max(float *d_outputBMag,float *d_outputBAng,
					float *d_outputGMag,float *d_outputGAng,
					float *d_outputRMag,float *d_outputRAng,
					float *d_outputMag,float  *d_outputAng,
					int imgRows, int  imgCols);