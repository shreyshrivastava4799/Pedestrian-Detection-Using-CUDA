/*  Host main routine   */
#include "headers.h"

using namespace cv;
using namespace std;

#define XIn 8
#define YIn 16

float* calcHistogram(float *magnitude, float*direction, int rows, int cols)
{

    cudaError_t err = cudaSuccess;

    size_t imageSize = rows*cols*sizeof(float);

    //final output array for feature vector of size 9*number of blocks
    float *final=(float*)malloc((9*imageSize/64));

    //creating the device array for the same
    float *d_hist_array=NULL;
    err = cudaMalloc((void **)&d_hist_array,(9*imageSize/64));

    
    //creating the device array for magnitude
    float *d_magnitude = NULL;
    err = cudaMalloc((void **)&d_magnitude, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for magnitude (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //copying the magnitude array into device magnitude array
    cudaMemcpy(d_magnitude,magnitude, imageSize, cudaMemcpyHostToDevice);
    
    
    //creating the device array for direction
    float *d_direction = NULL;
    err = cudaMalloc((void **)&d_direction, imageSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //copying the direction array into device direction array
    cudaMemcpy(d_direction,direction, imageSize, cudaMemcpyHostToDevice);
    
   
    //Specifying number of blocks and number of threads.
    dim3 grid(cols/8,rows/8,1);
    dim3 block(8,8,1);
    
    //calling the kernel
    histogram<<<grid,block>>>(d_magnitude,d_direction,rows,cols,d_hist_array);
   
    //copying the device array to host
    cudaMemcpy(final, d_hist_array, (9*imageSize/64), cudaMemcpyDeviceToHost);

    err = cudaFree(d_hist_array);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_magnitude);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_direction);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Calculation of histogram done.\n");
    
    return(final);
}

int *typecastHistograms(float *histIn)
{
	int numElements = XIn*YIn*9;
    size_t size = numElements * sizeof(int);
	int *histOut = (int *)malloc(size);

	for(int i=0; i<numElements; ++i)
	{
		*(histOut + i) = (int)(*(histIn + i));
	}
	return histOut;
}

float *calcNormalHist(int *h_Hist, size_t sizeIn)
{
    bool DEBUG = false;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Compute the size of normalized histogram
    int numElementsOut = (XIn - 1)*(YIn - 1)*36;
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
    if(DEBUG) 
        printf("Copy input data from the host memory to the CUDA device\n");
    
    err = cudaMemcpy(d_Hist, h_Hist, sizeIn, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the L2 Norm CUDA Kernel
    dim3 threadsPerBlock(XIn,YIn,1);
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
    if(DEBUG) 
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

	printf("Calculation of normalized histogram done.\n");
	
	// Return result
	return h_HistNorm;
}


int main(void)
{
    bool DEBUG = false;
    printf("Inside Host Code\n");
    
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    /*   Image Loading   */

    // OpenCV code for reading image
    Mat origImg = imread("../persons/person_024.bmp",1);


    // To verify if original image is loaded properly 
    if(DEBUG)
    {        
        imshow("PersonImage",origImg);
        waitKey(0);
    }  

    // resizing image
    int resolution = 2;
    int resizedRows  = origImg.rows/resolution;   
    int resizedCols  = origImg.cols/resolution;
    Mat resizeImg(resizedRows, resizedCols, CV_8UC3, Scalar(0,0,0));

    for (int k = 0; k < resizedRows; ++k)
    {
        for (int l = 0; l < resizedCols; ++l)
        {
            resizeImg.at<Vec3b>(k,l)[0] = origImg.at<Vec3b>(k*resolution,l*resolution)[0];
            resizeImg.at<Vec3b>(k,l)[1] = origImg.at<Vec3b>(k*resolution,l*resolution)[1];
            resizeImg.at<Vec3b>(k,l)[2] = origImg.at<Vec3b>(k*resolution,l*resolution)[2];

        }
    }   

    imshow("ResizedImage",resizeImg);
    waitKey(0);
    
    int windowsRows = 128;
    int windowsCols = 64;

    // Step of each window
    int StepSlide = 32;

    for (int row = 0; row < resizeImg.rows - windowsRows; row += StepSlide)
        for (int col = 0; col < resizeImg.cols - windowsCols; col += StepSlide)
        {
            
            Mat img(windowsRows, windowsCols, CV_8UC3, Scalar(0,0,0));        
            for (int k = 0; k < windowsRows; ++k)
            {
                for (int l = 0; l < windowsCols; ++l)
                {
                    img.at<Vec3b>(k,l)[0] = resizeImg.at<Vec3b>(row+k,col+l)[0];
                    img.at<Vec3b>(k,l)[1] = resizeImg.at<Vec3b>(row+k,col+l)[1];
                    img.at<Vec3b>(k,l)[2] = resizeImg.at<Vec3b>(row+k,col+l)[2];
                }
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
            if(DEBUG) 
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
            float *d_outputBMag = NULL;
            err = cudaMalloc((void **)&d_outputBMag, imageSize);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            float *d_outputBAng = NULL;
            err = cudaMalloc((void **)&d_outputBAng, imageSize);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device memory for output
            float *d_outputGMag = NULL;
            err = cudaMalloc((void **)&d_outputGMag, imageSize);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            float *d_outputGAng = NULL;
            err = cudaMalloc((void **)&d_outputGAng, imageSize);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Allocate the device memory for output
            float *d_outputRMag = NULL;
            err = cudaMalloc((void **)&d_outputRMag, imageSize);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            float *d_outputRAng = NULL;
            err = cudaMalloc((void **)&d_outputRAng, imageSize);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device memory for image (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

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
            convolution<<<X, Y, tileSize>>>(d_B, paddedX, paddedY, blockX, blockY, d_outputBMag, d_outputBAng, img.rows, img.cols);
            convolution<<<X, Y, tileSize>>>(d_G, paddedX, paddedY, blockX, blockY, d_outputGMag, d_outputGAng, img.rows, img.cols);
            convolution<<<X, Y, tileSize>>>(d_R, paddedX, paddedY, blockX, blockY, d_outputRMag, d_outputRAng, img.rows, img.cols);
           

            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch convolution kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            cudaDeviceSynchronize();

            Y.x = blockX, Y.y = blockY;
            max<<<X, Y>>>(d_outputBMag, d_outputBAng, d_outputGMag, d_outputGAng, d_outputRMag, d_outputRAng,
                d_outputMag, d_outputAng, img.rows, img.cols);
            
            // Copy the device result vector in device memory to the host result vector in host memory.
            if(DEBUG)     
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
            
            float* histGrad = calcHistogram(h_outputMag,h_outputAng,img.rows,img.cols);
                 
            
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
           

            // Free device global memory
            err = cudaFree(d_outputBMag);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free device array B (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaFree(d_outputBAng);
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


        	// Size of input vector
        	int numElementsIn = XIn*YIn*9;
        	size_t sizeIn = numElementsIn * sizeof(int);
        	
        	// Calculate final feature vector from HOG
        	int *histOutput = typecastHistograms(histGrad);
        	float *featureVec = calcNormalHist(histOutput, sizeIn);

            // // Verification of feature vector:
            // int numElementsOut = (XIn - 1)*(YIn - 1)*36;
            // for (int i = 0; i < numElementsOut; ++i)
            // {
            //     cout<<"i: "<<i<<" featureVec:"<<featureVec[i]<<endl;
            // }

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
        }    
        return 0;
}

