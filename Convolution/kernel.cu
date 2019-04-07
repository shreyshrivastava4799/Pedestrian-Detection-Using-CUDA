#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265
#define PADDING_SIZE 1

// declaring constant memory for kernel
__device__ __constant__ float d_filterKernel[FILTER_SIZE] = { -1, 0, 1};


__global__ void convolution( float *image, int paddedX, int paddedY,
							 int blockX, int blockY, 
							 float *outputMag, float *outputAng, int imgRows, int imgCols)
{

	/*
		this blockOrigin(X,Y) are w.r.t to the padded image 
		these are the coordinates where blocks (0,0) are placed in padded image
	*/
	
	int blockOriginX = blockIdx.x * blockX  + PADDING_SIZE;
	int blockOriginY = blockIdx.y * blockY  + PADDING_SIZE;
	// printf("blockOriginX:%d blockOriginY:%d\n",blockOriginX,blockOriginY);
	
	/*
		this tileOrigin(X,Y) are w.r.t to the padded image 
		these are the coordinates where tiles (0,0) are placed in padded image	
	*/	

	int tileOriginX = blockOriginX - PADDING_SIZE;
	int tileOriginY = blockOriginY - PADDING_SIZE;
	// printf("tileOriginX:%d tileOriginY:%d\n",tileOriginX,tileOriginY);

	/*	
		these coordinates specify the ends of this tile
		if( tileX is not divisible by 4 ) the last iteration when threads will be reused will exceed tile size
		so less than this tilesize or if ending of image is hit
		
	*/

	int tileEndX = min(tileOriginX + blockX + 2*PADDING_SIZE, paddedX);
	int tileEndY = min(tileOriginY + blockY + 2*PADDING_SIZE, paddedY);
	// printf("tileEndX:%d \n",tileEndX );
	
	// Allocating shared memory for this kernel
	extern __shared__ float imageTile[];

	/*
		the same thread is used to bring as many as 4 global memory to shared memory depending on whether it hits any boundary
	*/	

	for (int m = 0; m < 4; ++m)
	{
		/*		
			((m*blockDim.x)+threadIdx.x)) will give the threadIdx of tile along x direction in mth iteration 
			(tileOriginX +((m*blockDim.x)+threadIdx.x)) will give the global threadIdx along x direction in mth direction
		*/

		if( (tileOriginX +((m*blockDim.x)+threadIdx.x))<tileEndX &&  tileOriginY +threadIdx.y<tileEndY )
		{
				imageTile[((m*blockDim.x)+threadIdx.x)*blockDim.y+threadIdx.y] 
					= image[(tileOriginX +((m*blockDim.x)+threadIdx.x))*paddedY + tileOriginY  +threadIdx.y];		
		}
		// printf("threadIdx.x:%d threadIdx.y:%d \n",(m*blockDim.x)+threadIdx.x, threadIdx.y );

	}

	__syncthreads();

	for (int m = 0; m < 4; ++m)
	{
		if( ((m*blockDim.x)+threadIdx.x)<blockX && threadIdx.y<blockY 
			&& (tileOriginX + ((m*blockDim.x)+threadIdx.x))<imgRows && tileOriginY + threadIdx.y<imgCols)
		{
			double gX = 0, gY = 0, gMag = 0, gAng = 0;				
			for (int k = -PADDING_SIZE; k <= PADDING_SIZE ; ++k)
			{
				// along x direction 
				gX += d_filterKernel[k+PADDING_SIZE] * imageTile[((m*blockDim.x)+threadIdx.x +k +PADDING_SIZE )*blockDim.y + threadIdx.y +PADDING_SIZE];
				gY += d_filterKernel[k+PADDING_SIZE] * imageTile[((m*blockDim.x)+threadIdx.x +PADDING_SIZE )*blockDim.y + k +threadIdx.y +PADDING_SIZE];

			}
			// this 90.1 is to make gAng + else it becomes -0.0000
			gMag = sqrt(gX*gX + gY*gY);
			gAng = atan(gY/gX)*180.0/PI + 90.1;

			outputMag[(tileOriginX + ((m*blockDim.x)+threadIdx.x))*(imgCols)+ tileOriginY + threadIdx.y] = gMag;
			outputAng[(tileOriginX + ((m*blockDim.x)+threadIdx.x))*(imgCols)+ tileOriginY + threadIdx.y] = gAng;
			
			// printf("threadIdx.x:%d threadIdx.y:%d \n", (m*blockDim.x)+threadIdx.x, threadIdx.y );

		}
	}

}