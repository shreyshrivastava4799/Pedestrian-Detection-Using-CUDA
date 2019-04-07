#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void l2norm(const int*, float*);

#define s4XIn 8
#define s4YIn 16 

float *step4(int *h_Hist, size_t sizeIn)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Compute the size of normalized histogram
    int numElementsOut = (s4XIn - 1)*(s4YIn - 1)*36;
    size_t sizeOut = numElementsOut * sizeof(float);
    
    // Allocate the host output
    float *h_HistNorm = (float *)malloc(sizeOut);

    // Verify that allocations succeeded
    if (h_HistNorm == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate the device input
    int *d_Hist = NULL;
    err = cudaMalloc((void **)&d_Hist, sizeIn);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output
    float *d_HistNorm = NULL;
    err = cudaMalloc((void **)&d_HistNorm, sizeOut);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input in host memory to the device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_Hist, h_Hist, sizeIn, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the L2 Norm CUDA Kernel
    dim3 threadsPerBlock(s4XIn,s4YIn,1);
	dim3 blocksPerGrid(1,1,1);
	l2norm<<<blocksPerGrid, threadsPerBlock>>>(d_Hist, d_HistNorm);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch L2 Norm kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();

    // Copy the device result in device memory to the host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_HistNorm, d_HistNorm, sizeOut, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Free device memory
	err = cudaFree(d_Hist);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_Hist (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	
	err = cudaFree(d_HistNorm);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_HistNorm (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	printf("Step 4 done\n");
	
	// Return result
	return h_HistNorm;
}

int main()
{
	// Compute the size of histogram to be used
    int numElementsIn = s4XIn*s4YIn*9;
    size_t sizeIn = numElementsIn * sizeof(int);
    
    // Allocate the host input
    int *h_Hist = (int *)malloc(sizeIn);

    // Verify that allocations succeeded
    if (h_Hist == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input
    for (int i = 0; i < numElementsIn; ++i)
    {
        *(h_Hist + i)  = rand()%256;
    }
	
	// Begin step 4
	step4(h_Hist, sizeIn);

	return 0;
}


