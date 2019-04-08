#include "headers.h"

int main(void)
{
    int rows=128;
    int cols=64;
    int i=0;
 
    cudaError_t err = cudaSuccess;
    
    //magnitude array of size=number of pixels in the image
    float *magnitude=(float *)malloc(rows*cols*sizeof(float));

    //direction array of size=number of pixels in the image
    float *direction=(float *)malloc(rows*cols*sizeof(float));

    //Random values given to direction and magnitude array
    for(i=0;i<rows*cols;++i)
    {
            magnitude[i]=100.5;
            
            if(i%2==1)
               direction[i]=90;
            
            else
               direction[i]=96;
    }
    
    //final output array for feature vector of size 9*number of blocks
    float *final=(float*)malloc((9*rows*cols/64)*sizeof(float));

    //creating the device array for the same
    float *d_hist_array=NULL;
    err = cudaMalloc((void **)&d_hist_array,(9*rows*cols/64)*sizeof(float));

    
    //creating the device array for magnitude
    float *d_magnitude = NULL;
    err = cudaMalloc((void **)&d_magnitude, rows*cols*sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for magnitude (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //copying the magnitude array into device magnitude array
    cudaMemcpy(d_magnitude,magnitude, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    
    
    //creating the device array for direction
    float *d_direction = NULL;
    err = cudaMalloc((void **)&d_direction, rows*cols*sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //copying the direction array into device direction array
    cudaMemcpy(d_direction,direction, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    
   
    //Specifying number of blocks and number of threads.
    dim3 grid(cols/8,rows/8,1);
    dim3 block(8,8,1);
    
     
    
    histogram<<<grid,block>>>(d_magnitude,d_direction,rows,cols,d_hist_array);
   
    cudaMemcpy(final, d_hist_array, (9*rows*cols/64)*sizeof(float), cudaMemcpyDeviceToHost);
    
    for(i=0;i<(9*rows*cols/64);++i)
    {
        cout<<"i= "<<i<<" "<<final[i]<<" blocknum="<<i/9<<endl;
    }


}

