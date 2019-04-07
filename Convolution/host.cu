/*  Host main routine   */
#include "headers.h"

using namespace cv;
using namespace std;


int main(void)
{
    bool DEBUG = true;
    printf("Inside Host Code\n");
    
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    /*   Image Loading   */

    // OpenCV code for reading image
    Mat img = imread("../persons/person_024.bmp",1);

    // To verify if original image is loaded properly 
    if(DEBUG)
    {        
        imshow("PersonImage",img);
        waitKey(0);
    }     
   
    // Padding required depending on kernel size
    // here kernel size is fixed always as 1 so 2*1
    int padding = 2;

    // Providing padding to image
    // X will be treated as rows and Y as cols
    int paddedX = img.rows + padding;
    int paddedY = img.cols + padding;

    size_t imageSize = img.rows * img.cols * sizeof(float);
    size_t paddedImageSize = paddedX * paddedY * sizeof(float);

    // Allocate memory for Blue Channel of image
    float *h_B = (float *)malloc(paddedImageSize);

    // Allocate memory for Green Channel of image
    float *h_G = (float *)malloc(paddedImageSize);

    // Allocate memory for Red Channel of image
    float *h_R = (float *)malloc(paddedImageSize);

    // Verify that allocations succeeded
    if (h_B == NULL || h_G == NULL || h_R == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for image!\n");
        exit(EXIT_FAILURE);
    }

    // Converting Mat to 1D array 
    for (int i = 0; i < paddedX; ++i)
        for (int j = 0; j < paddedY; ++j)
        {
            if( i==0 || i==paddedX-1 || j==0 || j==paddedY-1 )
            {
                h_B[i*paddedY + j] = 0;
                h_G[i*paddedY + j] = 0;
                h_R[i*paddedY + j] = 0;
            }                
            else
            {
                h_B[i*paddedY + j] = img.at<Vec3b>(i,j)[0];
                h_G[i*paddedY + j] = img.at<Vec3b>(i,j)[1];
                h_R[i*paddedY + j] = img.at<Vec3b>(i,j)[2];
            }
        }


    // Verify that the channel array is correct
    if(DEBUG)
    {   
        Mat checkImage(paddedX,paddedY, CV_8UC1, Scalar(0));
        for (int i = 0; i < paddedX*paddedY; ++i)
        {
            checkImage.at<uchar>(i/paddedY,i%paddedY) = h_B[i];
        }
        imshow("checkImage", checkImage);
        waitKey(0);
    }

    
    // Allocate the device memory for Blue Channel
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, paddedImageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for B channel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device memory for Green Channel
    float *d_G = NULL;
    err = cudaMalloc((void **)&d_G, paddedImageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Allocate the device memory for Red Channel
    float *d_R = NULL;
    err = cudaMalloc((void **)&d_R, paddedImageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //  Copy image from host memory to device memory 
    printf("Copying image from host memory to device memory.\n");
    err = cudaMemcpy(d_B, h_B, paddedImageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy blue channel image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_G, h_G, paddedImageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy green channel image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_R, h_R, paddedImageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy red channel image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate memory for output gradient values: magnitude and angle
    float *h_outputMag = (float *)malloc(imageSize);
    float *h_outputAng = (float *)malloc(imageSize);

    // Allocate the device memory for output
    float *d_outputMag = NULL;
    err = cudaMalloc((void **)&d_outputMag, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_outputAng = NULL;
    err = cudaMalloc((void **)&d_outputAng, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Image is divided in no of image blocks gradients for each will be calculated parallely
    // Size of image block that will have its gradient calc. in one kernel call
    int blockX = 32, blockY = 32;

    // Size to be allocated for shared memory inside kernel
    // This is the size of block along with padding so that convolution can be done 
    // at the border points
    int tileX = blockX + padding;
    int tileY = blockY + padding; 
    size_t tileSize = (tileX)*(tileY)*sizeof(float);

    // for each tile only 4th the threads are allocated and then reused accordingly
    int blockDimX = ceil((double)tileX/4), blockDimY = tileY;

    // the no. of thread blocks that have to be launched will be the no. of image rows and cols 
    // divided by the no. of pixel we wish to keep in one block of image
    int gridDimX = ceil((double)img.rows/blockX), gridDimY = ceil((double)img.cols/blockY);

    // Specifying execution configuration
    dim3 X(gridDimX,gridDimY);
    dim3 Y(blockDimX,blockDimY);
    convolution<<<X, Y, tileSize>>>(d_B, paddedX, paddedY, blockX, blockY, d_outputMag, d_outputAng, img.rows, img.cols);
   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convolution kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_outputMag, d_outputMag, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_outputAng, d_outputAng, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the resulting image is correct
    Mat magImage(img.rows, img.cols, CV_8UC1, Scalar(0));
    Mat angleImage(img.rows, img.cols, CV_8UC1, Scalar(0));
    for (int i = 0; i < img.rows*img.cols; ++i)
    {
        magImage.at<uchar>(i/img.cols,i%img.cols) = h_outputMag[i];
        angleImage.at<uchar>(i/img.cols,i%img.cols) = h_outputAng[i];
    }
    imshow("Output Angle", angleImage);
    imshow("Output Maginitude", magImage);
    waitKey(0);
   


    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_outputMag);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_outputAng);
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

