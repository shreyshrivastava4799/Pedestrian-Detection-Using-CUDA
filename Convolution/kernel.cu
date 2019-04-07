#include <stdio.h>
#define PADDING_SIZE 1
#define FILTER_SIZE 3

__device__ __constant__ float d_filterKernel[FILTER_SIZE] = { -1, 0, 1};

// This convolution kernel calculates gradient only along rows
__global__ void convolution( float *image, int paddedX, int paddedY,
							 int blockX, int blockY, float *output	)
{

	// this blockX and blockY should be wrt to original image 
	// so that we can have tiles just as an external abstraction in 
	// our kernel

	// finding what out (0,0) in block means globally 
	// we have added PADDING_SIZE because the image passed is already padded in that think about (0,0)
	// of the image where it will start
	int blockOriginX = blockIdx.x * blockX  + PADDING_SIZE;
	int blockOriginY = blockIdx.y * blockY  + PADDING_SIZE;
	// printf("blockOriginX:%d blockOriginY:%d\n",blockOriginX,blockOriginY);

	// finding what out (0,0) of tile globally
	int tileOriginX = blockOriginX - PADDING_SIZE;
	int tileOriginY = blockOriginY - PADDING_SIZE;
	// printf("tileOriginX:%d tileOriginY:%d\n",tileOriginX,tileOriginY);

	// Allocating shared memory for this kernel
	extern __shared__ float imageTile[];

	for (int m = 0; m < 4; ++m)
	{
		// if( (tileOriginX +((m*blockDim.x)+threadIdx.x))<paddedX &&  tileOriginY +threadIdx.y<paddedY )
		// {
				imageTile[((m*blockDim.x)+threadIdx.x)*blockDim.y+threadIdx.y] 
					= image[(tileOriginX +((m*blockDim.x)+threadIdx.x))*paddedY + tileOriginY  +threadIdx.y];
				// printf("Shared Memory:%f Image Pixel:%f\n",imageTile[((m*blockDim.x)+threadIdx.x)*blockDim.y+ threadIdx.y],image[(tileOriginX +((m*blockDim.x)+threadIdx.x))*paddedY+tileOriginY+threadIdx.y] );	
				// printf("tile_x:%d tile_y:%d global_x:%d global_y:%d tileOriginx:%d tileOriginY:%d \n",(m*blockDim.x)+threadIdx.x
				// 	,threadIdx.y,(tileOriginX +((m*blockDim.x)+threadIdx.x)), tileOriginY  +threadIdx.y, tileOriginX, tileOriginY);
		// }

	}

	__syncthreads();

	for (int m = 0; m < 4; ++m)
	{
		// float temp = 0;
		// if( blockX+threadIdx.x<paddedX-PADDING_SIZE && blockY+threadIdx.y< paddedY-PADDING_SIZE )
		// {		
		// 	// for (int k = -PADDING_SIZE; k <= PADDING_SIZE ; ++k)
		// 	// {
		// 	// 	// along x direction 
		// 	// 	temp += d_filterKernel[k+PADDING_SIZE] * imageTile[(m+1)*(threadIdx.x+k)*blockDim.y + threadIdx.y];
		// 	// }
		// }
		// if( ((m*blockDim.x)+threadIdx.x)>0 && ((m*blockDim.x)+threadIdx.x)<blockDim.x*4 && threadIdx.y>0 && threadIdx.y < blockDim.y-1 )
		if( ((m*blockDim.x)+threadIdx.x)<blockDim.x*4-2 && threadIdx.y<blockDim.y-2 )
		{
			// printf("threadIdx.x:%d threadIdx.y:%d \n",(m*blockDim.x)+threadIdx.x, threadIdx.y );
			output[(blockOriginX + ((m*blockDim.x)+threadIdx.x))*(paddedY-2*PADDING_SIZE)+ blockOriginY + threadIdx.y] 
			= imageTile[((m*blockDim.x)+threadIdx.x +PADDING_SIZE)*blockDim.y + threadIdx.y + PADDING_SIZE];
		}
	}

}