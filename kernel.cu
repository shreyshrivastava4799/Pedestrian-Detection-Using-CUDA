#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265
#define PADDING_SIZE 1
#define FILTER_SIZE 3

#define X 8
#define Y 16

// declaring constant memory for kernel
__device__ __constant__ float d_filterKernel[FILTER_SIZE] = { -1, 0, 1};

__global__ void convolutionGlobal( float *image, int height, int width,
							 	   float *outputMag, float *outputAng	)
{
	/*
		this blockOrigin(X,Y) are w.r.t to the padded image 
		these are the coordinates where blocks (0,0) are placed in padded image
	*/
	
	int blockOriginX = blockIdx.x * blockDim.x  + PADDING_SIZE;
	int blockOriginY = blockIdx.y * blockDim.y  + PADDING_SIZE;
	// printf("blockOriginX:%d blockOriginY:%d\n",blockOriginX,blockOriginY);
	
	/*
		this tileOrigin(X,Y) are w.r.t to the padded image 
		these are the coordinates where tiles (0,0) are placed in padded image	
	*/	

	int tileOriginX = blockOriginX - PADDING_SIZE;
	int tileOriginY = blockOriginY - PADDING_SIZE;
	// printf("tileOriginX:%d tileOriginY:%d\n",tileOriginX,tileOriginY);

	int pixelX = blockOriginX + threadIdx.x;
	int pixelY = blockOriginY + threadIdx.y;

	if( (tileOriginX + threadIdx.x)< height - 2*PADDING_SIZE && tileOriginY + threadIdx.y < width - 2*PADDING_SIZE )
	{	
		double gX = 0, gY = 0, gMag = 0, gAng = 0;				
		for (int k = -PADDING_SIZE; k <= PADDING_SIZE ; ++k)
		{
			// along x direction 
			gX += d_filterKernel[k+PADDING_SIZE] * image[(pixelX + k)*width + (pixelY)];
			gY += d_filterKernel[k+PADDING_SIZE] * image[(pixelX)*width + (pixelY+k)];

		}
		// this 90.1 is to make gAng + else it becomes -0.0000
		gMag = sqrt(gX*gX + gY*gY);
		
		if( gX==0 )
			gAng = 90;
		else
			gAng = atan(gY/gX)*180.0/PI + 90.1;

		outputMag[(tileOriginX + threadIdx.x)*(width-2*PADDING_SIZE)+ tileOriginY + threadIdx.y] = gMag;
		outputAng[(tileOriginX + threadIdx.x)*(width-2*PADDING_SIZE)+ tileOriginY + threadIdx.y] = gAng;	
	}
}

__global__ void convolutionShared(float *image, int paddedX, int paddedY,
   								  int blockX, int blockY, 
   								  float *outputMag, float *outputAng,
   								  int imgRows, int imgCols)
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
			if( gX==0 )
				gAng = 90;
			else 
				gAng = atan(gY/gX)*180.0/PI + 90.1;
			outputMag[(tileOriginX + ((m*blockDim.x)+threadIdx.x))*(imgCols)+ tileOriginY + threadIdx.y] = gMag;
			outputAng[(tileOriginX + ((m*blockDim.x)+threadIdx.x))*(imgCols)+ tileOriginY + threadIdx.y] = gAng;
			
			// printf("threadIdx.x:%d threadIdx.y:%d \n", (m*blockDim.x)+threadIdx.x, threadIdx.y );

		}
	}

}

__global__ void max(float *d_outputBMag,float *d_outputBAng,
					float *d_outputGMag,float *d_outputGAng,
					float *d_outputRMag,float *d_outputRAng,
					float *d_outputMag,float  *d_outputAng,
					int imgRows, int  imgCols)
{

	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if( tidx<imgRows && tidy<imgCols)
	{
		float maxMag = d_outputBMag[tidx*imgCols + tidy], maxAng = d_outputBAng[tidx*imgCols + tidy];
		
		if( maxMag < d_outputGMag[tidx*imgCols + tidy] )
			maxMag = d_outputGMag[tidx*imgCols + tidy], maxAng = d_outputGAng[tidx*imgCols + tidy];

		if( maxMag < d_outputRMag[tidx*imgCols + tidy] )
			maxMag = d_outputRMag[tidx*imgCols + tidy], maxAng = d_outputRAng[tidx*imgCols + tidy];

		d_outputMag[tidx*imgCols + tidy] = maxMag,d_outputAng[tidx*imgCols + tidy] = maxAng; 
	}


}


__global__ void histogram(float *mag,float *dir, int height, int width,float *output)
{
   
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    float magnitude = mag[i*width+j];
    float direction = dir[i*width+j];

    int blockNum = blockIdx.y*gridDim.x + blockIdx.x;

    // initializing the 9-element array for each block.
    if( threadIdx.x==0 && threadIdx.y==0 )
    {
	    for(int k=0;k<9;++k)
	    	output[k+blockNum*9]=0;
	}

    // waiting for initialization.
    __syncthreads();

    // calculating the histogram values
    // calculating the lower angle container for this gradient value 
    int low = (direction/20);
    atomicAdd(&output[blockNum*9+low],magnitude*((low+1)*20 -direction)/20.0);
    atomicAdd(&output[blockNum*9+(low+1)%9],magnitude*((direction-low*20)/20.0));	
}


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
			*(output + 36*(y*(X-1) + x) + i + 9) = *(hist + 9*(y*X + x + 1) + i)/norm;
			*(output + 36*(y*(X-1) + x) + i + 18) = *(hist + 9*((y + 1)*X + x + 1) + i)/norm;
			*(output + 36*(y*(X-1) + x) + i + 27) = *(hist + 9*((y + 1)*X + x) + i)/norm;
		}
	}
}

__global__ void LinearSVMEvaluation(float *inputs, float *weigths, float bias,
                                    int blockSizeX, int blockSizeY, int numBlocksPerWindowX,
                                    int numBlocksPerWindowY, float *svmScores
                                      )
{
	// int numBlocksX = 1;

	int col = threadIdx.x;
	int totalCols = blockDim.x;
	//int imWidth = blockSizeX * numBlocksX;
	//int WinOff = blockIdx.x * blockSizeX + blockIdx.y * blockSizeY * blockSizeX;
	__shared__ float sum[18*7];
	int i;

	//multiply features by their respective weights parallely.
	for(i = 0; i < numBlocksPerWindowY * blockSizeY; i++){
	sum[col] = inputs[i * totalCols + col] * weigths[i * totalCols + col];
	__syncthreads();
	}

	//parallel reduction.
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(col < s){
			sum[col] += sum[col + s];
		}
		__syncthreads();
	}

	//subtract bias and store final score in global memory.
	if(col==0){
	sum[0] -= bias;
	svmScores[0] = sum[0];
}

}