/*  Host main routine   */
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void LinearSVMEvaluation(float *, float *, float , int ,
                                    int , int , int ,
                                    int ,
                                    float *
                                  );

int main(void)
{

      int winsize = 3780, blockSizeX = 18, blockSizeY = 2;
      int numBlocksPerWindowX = 7, numBlocksPerWindowY = 15;
      cudaError_t err = cudaSuccess;
      float *h_weights = (float *)malloc(winsize*sizeof(float));
      float bias;

      float *inputs= NULL;
      float final[] = {1.0,2.0,1.0,1.0};
      err = cudaMalloc((void **)&inputs,winsize*sizeof(float));
      cudaMemcpy(inputs, final, winsize*sizeof(float), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
    	{
        fprintf(stderr, "Failed to allocate device memory for magnitude (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    	}

      float *d_weights=NULL;
      cudaMalloc((void **)&d_weights,(winsize*sizeof(float)));
      //printf("%d",err);

      //int opX = (numBlocksX - numBlocksPerWindowX) + 1;
      //int opY = (numBlocksY - numBlocksPerWindowY) + 1;
      int opX = 1, opY = 1;
      float *h_svmScores = (float *)malloc(opX*opY*sizeof(float));

      float *d_svmScores = NULL;
      cudaMalloc((void **)&d_svmScores,(opX*opY*sizeof(float)));
      

      FILE *f = fopen("svmweights.txt","r");
      for(int i = 0; i < winsize; i++){
          fscanf(f, "%f", h_weights+i);
          //printf("%f\t", final[i]);
      }
      
      fscanf(f, "%f", &bias);
      //printf("%f\n", bias);
    
      err = cudaMemcpy(d_weights, h_weights, winsize*sizeof(float), cudaMemcpyHostToDevice);
      if(err != cudaSuccess)
      	printf("error2\n");

      dim3 grid1(1,1,1);
      dim3 block1(numBlocksPerWindowX*blockSizeX , 1 ,1);
      printf("%d\n", numBlocksPerWindowX*blockSizeX);
      
      LinearSVMEvaluation<<<grid1, block1>>>(inputs, d_weights, bias, 0,
        blockSizeX, blockSizeY, numBlocksPerWindowX, numBlocksPerWindowY, d_svmScores);

      err = cudaMemcpy(h_svmScores, d_svmScores, opX*opY*sizeof(float), cudaMemcpyDeviceToHost);
      if(err != cudaSuccess)
      	printf("error3\n");

      printf("%f",h_svmScores[0]);

    return 0;

}
