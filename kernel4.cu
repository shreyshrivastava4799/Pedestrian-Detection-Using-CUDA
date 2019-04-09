#include <math.h>

#define X 8
#define Y 16

__global__
void l2norm(const int *input, float *output)
{
	// Shared memory for kernel
	__shared__ int hist[Y*X*9];

	// Index for 16x16 window
	int x = threadIdx.x;
	int y = threadIdx.y;

	// Copy the top left 8x8 block to shared memory
	for(int i=0; i<9; ++i)
	{
		*(hist + 9*(y*X + x) + i) = *(input + 9*(y*X + x) + i);
	}

	// Synchronize threads after all shared memory is copied
	__syncthreads();

	// Normalize the 36 length feature vector
	if(x != X-1 && y != Y-1)
	{
		// Calculate the normalizing factor for 16x16 window
		float norm = 0;
		for(int i=0; i<9; ++i)
		{
			norm += powf(*(hist + 9*(y*X + x) + i), 2);
			norm += powf(*(hist + 9*(y*X + x + 1) + i), 2);
			norm += powf(*(hist + 9*((y + 1)*X + x + 1) + i), 2);
			norm += powf(*(hist + 9*((y + 1)*X + x) + i), 2);
		}
		norm = sqrt(norm);

		// Normalize and store the output feature vector
		for(int i=0; i<9; ++i)
		{
			*(output + 36*(y*(X-1) + x) + i) = *(hist + 9*(y*X + x) + i)/norm;
		}
		for(int i=0; i<9; ++i)
		{
			*(output + 36*(y*(X-1) + x) + i + 9) = *(hist + 9*(y*X + x + 1) + i)/norm;
		}
		for(int i=0; i<9; ++i)
		{
			*(output + 36*(y*(X-1) + x) + i + 18) = *(hist + 9*((y + 1)*X + x + 1) + i)/norm;
		}
		for(int i=0; i<9; ++i)
		{
			*(output + 36*(y*(X-1) + x) + i + 27) = *(hist + 9*((y + 1)*X + x) + i)/norm;
		}
	}
}


