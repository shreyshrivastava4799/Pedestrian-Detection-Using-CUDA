#include <stdio.h>
#define PADDING_SIZE 1
#define FILTER_SIZE 3

__device__ __constant__ float d_filterKernel[FILTER_SIZE] = { -1, 0, 1};

// This convolution kernel calculates gradient only along rows
__global__ void convolution( float *image, int paddedX, int paddedY,
							 int blockX, int blockY, float *output	)
{
	// bool DEBUG = true;
	
	// if(DEBUG)

	// this blockX and blockY should be wrt to original image 
	// so that we can have tiles just as an external abstranction in 
	// out kernel

	// finding what out (0,0) in block means globally 
	// we have added PADDING_SIZE because the image passed is already padded in that think about (0,0)
	int blockOriginX = blockIdx.x * blockX  + PADDING_SIZE;
	int blockOriginY = blockIdx.y * blockY  + PADDING_SIZE;
	// printf("blockX:%d blockY:%d\n",blockX,blockY);

	// finding what out (0,0) in tile means globally 
	int tileOriginX = blockOriginX - PADDING_SIZE;
	int tileOriginY = blockOriginY - PADDING_SIZE;
	// printf("tileX:%d tileY:%d\n",tileX,tileY);

	// Allocating shared memory for this kernel
	extern __shared__ float imageTile[];

	// finding the particular pixel value globally each thread is assigned to 
	// we use threadIdx.x and threadIdx.y for local thread allocation
	// int pixelX = tileX + threadIdx.x;
	// int pixelY = tileY + threadIdx.y;

	// printf("TileX: %f TileY: %f threadIdx.x:%d threadIdx.y:%d blockX:%d blockY:%d\n",tileX,tileY,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
	// Using the same thread for fetching 4 different global values
	// imageTile[threadIdx.x*blockDim.y+ threadIdx.y] = image[tileX + threadIdx.x*paddedY+ tileY + threadIdx.y];
	// imageTile[((1*blockDim.x)+threadIdx.x)*blockDim.y+ threadIdx.y] = image[tileX + ((1*blockDim.x)+threadIdx.x)*paddedY+ tileY + threadIdx.y];
	// imageTile[((2*blockDim.x)+threadIdx.x)*blockDim.y+ threadIdx.y] = image[tileX + ((2*blockDim.x)+threadIdx.x)*paddedY+ tileY + threadIdx.y];
	// printf("pixelX:%d  pixelY:%d\n",((3*blockDim.x)+threadIdx.x), threadIdx.y);	


	for (int m = 0; m < 4; ++m)
	{
		if( (tileOriginX +((m*blockDim.x)+threadIdx.x))<paddedX &&  tileOriginY +threadIdx.y<paddedY )
		{
			imageTile[((m*blockDim.x)+threadIdx.x)*blockDim.y+ threadIdx.y] = image[(tileOriginX +((m*blockDim.x)+threadIdx.x))*paddedY + tileOriginY  +threadIdx.y];
			// printf("Shared Memory:%f Image Pixel:%f\n",imageTile[(m+1)*threadIdx.x*blockDim.y+ threadIdx.y], image[tileOriginX + (m+1)*threadIdx.x*blockDim.y+ tileOriginY  + threadIdx.y]);	
		}

	}
    // printf("threadIdx.x:%d threadIdx.y:%d blockX:%d blockY:%d\n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);

	__syncthreads();

	// printf("paddedX: %d paddedY: %d\n",paddedX, paddedY );

	// for (int i = 0; i < blockDim.x ; ++i)
	// {
	// 	for (int j = 0; j < blockDim.y; ++j)
	// 	{
	// 		printf("Shared Memory:%d Image Pixel:%d\n",imageTile[i*blockDim.y+j], image[i*paddedY+j]);	
	// 	}
	// }
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
		if( ((m*blockDim.x)+threadIdx.x)>0 && ((m*blockDim.x)+threadIdx.x)<blockDim.x*4 && threadIdx.y>0 && threadIdx.y < blockDim.y-1 )
		{
			output[blockOriginX + ((m*blockDim.x)+threadIdx.x)*(paddedY-2*PADDING_SIZE)+ blockOriginY + threadIdx.y] = imageTile[(((m*blockDim.x)+threadIdx.x))*blockDim.y + threadIdx.y];
		}
	}

}