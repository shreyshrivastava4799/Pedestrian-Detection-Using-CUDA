__global__ void LinearSVMEvaluation(float *inputs, float *weigths, float bias, int loc,
                                    int blockSizeX, int blockSizeY, int numBlocksPerWindowX,
                                    int numBlocksPerWindowY,
                                    float *svmScores
                                      )
{
  int col = threadIdx.x;
  int totalCols = blockDim.x;

  __shared__ float sum[4];
  int i;

  for(i = 0; i < numBlocksPerWindowY * blockSizeY; i++){
    sum[col] = inputs[i * totalCols + col] * weigths[i * totalCols + col];
    __syncthreads();
  }

  for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(col < s){
			sum[col] += sum[col + s];
		}
		__syncthreads();
	}


  if(col==0){
    sum[0] += bias;
    svmScores[loc] = sum[0];
  }

}
