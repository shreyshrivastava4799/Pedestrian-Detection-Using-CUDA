/*
  Host main routine
*/
#include "headers.h"

using namespace cv;
using namespace std;


int main(void)
{
    bool DEBUG = true;
    printf("Inside Host Code\n");
    
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    /*
        Kernel Loading 
    */

/*  
    int filterH, filterW;
    cout<<"Enter filter height and width:"<<endl;
    cin >> filterH >> filterW;
    
    // Print the vector length to be used, and compute its size
    int filterSize = filterH * filterW;
    size_t filterSizeInByte = filterSize * sizeof(float);

    // Allocate memory for filter kernel
    float *h_filterKernel = (float *)malloc(filterSizeInByte);

    // Initialize the host input vectors
    for (int i = 0; i < filterSize; ++i)
        cin>> h_filterKernel[i];

    // Transfer host data to constant device memory
    cudaMemcpyToSymbol( d_filterKernel, h_filterKernel, filterSizeInByte, 0,cudaMemcpyHostToDevice);
*/
    /*
        Image Loading 
    */

    // OpenCV code for reading image
    Mat img = imread("../persons/person_024.bmp",1);
   
    imshow("PersonImage",img);
    waitKey(0);
   
    // Padding required depending on kernel size
    int padding = 2;

    // Providing padding to image
    int paddedR = img.rows + padding;
    int paddedC = img.cols + padding;

    size_t imageSize = paddedR * paddedC * sizeof(float);

    // Allocate memory for Blue Channel of image
    float *h_B = (float *)malloc(imageSize);

    // Allocate memory for Green Channel of image
    float *h_G = (float *)malloc(imageSize);

    // Allocate memory for Red Channel of image
    float *h_R = (float *)malloc(imageSize);

    // Verify that allocations succeeded
    if (h_B == NULL || h_G == NULL || h_R == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for image!\n");
        exit(EXIT_FAILURE);
    }

    // Converting Mat to 1D array 
    for (int i = 0; i < paddedR; ++i)
        for (int j = 0; j < paddedC; ++j)
        {
            if( i==0 || i==paddedR-1 || j==0 || j==paddedC-1 )
            {
                h_B[i*paddedC + j] = 0;
                h_G[i*paddedC + j] = 0;
                h_R[i*paddedC + j] = 0;
            }                
            else
            {
                h_B[i*paddedC + j] = img.at<Vec3b>(i,j)[0];
                h_G[i*paddedC + j] = img.at<Vec3b>(i,j)[1];
                h_R[i*paddedC + j] = img.at<Vec3b>(i,j)[2];
            }
        }


    // Verify that the channel array is correct
    if(DEBUG)
    {   
        Mat checkImage(paddedR,paddedC, CV_8UC1, Scalar(0));
        for (int i = 0; i < paddedR*paddedC; ++i)
        {
            checkImage.at<uchar>(i/paddedC,i%paddedC) = h_B[i];
        }
        imshow("checkImage", checkImage);
        waitKey(0);
    }

    
    // Allocate the device memory for Blue Channel
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for B channel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device memory for Green Channel
    float *d_G = NULL;
    err = cudaMalloc((void **)&d_G, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Allocate the device memory for Red Channel
    float *d_R = NULL;
    err = cudaMalloc((void **)&d_R, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Copy image from host memory to device memory 
    printf("Copying image from host memory to device memory.\n");
    err = cudaMemcpy(d_B, h_B, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy blue channel image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_G, h_G, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy green channel image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_R, h_R, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy red channel image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    

    

    // Allocate the device memory for output
    float *d_output = NULL;
    // err = cudaMalloc((void **)&d_output, tileSize);
    err = cudaMalloc((void **)&d_output, img.rows*img.cols*sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Allocate memory output gradient values
    // float *h_output = (float *)malloc(tileSize);
    float *h_output = (float *)malloc(img.rows*img.cols*sizeof(float));

    // Size of image block that will have its gradient calc. in one kernel call
    int blockX = 30, blockY = 30;

    // Size to be allocated for shared memory inside kernel
    int tileX = blockX+padding;
    int tileY = blockY+padding; 
    size_t tileSize = (tileX)*(tileY)*sizeof(float);
    // Specifying execution configuration
    // cout<<"Verification: "<<ceil(img.rows/blockX)<<" "<<(img.cols/blockY)<<endl;
    dim3 X(ceil(img.rows/blockX),ceil(img.cols/blockY));
    // dim3 X(1,1);
    dim3 Y(tileX/4  ,tileY);

    convolution<<<X, Y, tileSize>>>(d_B, paddedR, paddedC, blockX, blockY, d_output);
   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convolution kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_output, d_output, tileSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    Mat featureImage(img.rows, img.cols, CV_8UC1, Scalar(0));
    // Verify that the result vector is correct
    for (int i = 0; i < img.rows*img.cols; ++i)
    {
        featureImage.at<uchar>(i/img.cols,i%img.cols) = h_output[i];
        // cout<<(int)featureImage.at<uchar>(i/tileY,i%tileY)<<endl;
    }
    // imshow("Input Image", origImage);
    imshow("Output Image", featureImage);
    waitKey(0);

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_G);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array G (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_R);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_B);
    free(h_G);
    free(h_R);

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

    printf("Done\n");
    return 0;
}

