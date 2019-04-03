#define PADDING_SIZE 1
#define FILTER_SIZE 3

__device__ __constant__ float d_filterKernel[FILTER_SIZE] = { -1, 0, 1};

// This convolution kernel calculates gradient only along rows
__global__ void convolution( float *image, int height, int width,
							 float *output	)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i< height && j<width)
	{	
		float temp = 0;
		for (int k = -PADDING_SIZE; k <= PADDING_SIZE ; ++k)
		{
			temp += d_filterKernel[k+PADDING_SIZE] * image[(i+k+PADDING_SIZE)*width + (j+PADDING_SIZE)];
		}
		output[i*width + j] = temp;
	}
}