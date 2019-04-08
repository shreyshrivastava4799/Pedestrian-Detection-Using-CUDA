__global__ void histogram(float *grad,float *dir, int height, int width,float *output)
{
    // 
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    float magnitude=grad[i*width+j];
    float direction=dir[i*width+j];

    int blockNum=blockIdx.y*gridDim.x+blockIdx.x;

    //initializing the 9-element array for each block.
    if(threadIdx.x==0&&threadIdx.y==0)
    {
	    for(int k=0;k<9;++k)
	    {
	    	output[k+blockNum*9]=0;
	    }
	}
    __syncthreads();
    //waiting for initialization.

    //calculating the histogram values
    int low=(direction/20);
    atomicAdd(&output[blockNum*9+low],magnitude*((low+1)*20 -direction)/20.0);
    atomicAdd(&output[blockNum*9+(low+1)%9],magnitude*((direction-low*20)/20.0));	
}
