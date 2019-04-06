__global__ void imageFilteringKernel( const T *d_f, const unsigned int paddedW, const unsigned int paddedH,
                      const unsigned int blockW, const unsigned int blockH, const int S,
                      T *d_h, const unsigned int W, const unsigned int H )
{

    //
    // Note that blockDim.(x,y) cannot be used instead of blockW and blockH,
    // because the size of a thread block is not equal to the size of a data block
    // due to the apron and the use of subblocks.
    //

    //
    // Set the size of a tile
    //
    const unsigned int tileW = blockW + 2 * S;
    const unsigned int tileH = blockH + 2 * S;

    // 
    // Set the number of subblocks in a tile
    //
    const unsigned int noSubBlocks = static_cast<unsigned int>(ceil( static_cast<double>(tileH)/static_cast<double>(blockDim.y) ));

    //
    // Set the start position of a data block, which is determined by blockIdx. 
    // Note that since padding is applied to the input image, the origin of the block is ( S, S )
    //
    const unsigned int blockStartCol = blockIdx.x * blockW + S;
    const unsigned int blockEndCol = blockStartCol + blockW;

    const unsigned int blockStartRow = blockIdx.y * blockH + S;
    const unsigned int blockEndRow = blockStartRow + blockH;

    //
    // Set the position of the tile which includes the data block and its apron
    //
    const unsigned int tileStartCol = blockStartCol - S;
    const unsigned int tileEndCol = blockEndCol + S;
    const unsigned int tileEndClampedCol = min( tileEndCol, paddedW );

    const unsigned int tileStartRow = blockStartRow - S;
    const unsigned int tileEndRow = blockEndRow + S;
    const unsigned int tileEndClampedRow = min( tileEndRow, paddedH );

    //
    // Set the size of the filter kernel
    //
    const unsigned int kernelSize = 2 * S + 1;

    //
    // Shared memory for the tile
    //
    extern __shared__ T sData[];

    //
    // Copy the tile into shared memory
    //
    unsigned int tilePixelPosCol = threadIdx.x;
    unsigned int iPixelPosCol = tileStartCol + tilePixelPosCol;
    for( unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++ ) {

        unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
        unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

        if( iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow ) { // Check if the pixel in the image
            unsigned int iPixelPos = iPixelPosRow * paddedW + iPixelPosCol;
            unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;
            sData[tilePixelPos] = d_f[iPixelPos];
        }

    }

    //
    // Wait for all the threads for data loading
    //
    __syncthreads();

    //
    // Perform convolution
    //
    tilePixelPosCol = threadIdx.x;
    iPixelPosCol = tileStartCol + tilePixelPosCol;
    for( unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++ ) {

        unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
        unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

        // Check if the pixel in the tile and image.
        // Note that the apron of the tile is excluded.
        if( iPixelPosCol >= tileStartCol + S && iPixelPosCol < tileEndClampedCol - S &&
            iPixelPosRow >= tileStartRow + S && iPixelPosRow < tileEndClampedRow - S ) {

            // Compute the pixel position for the output image
            unsigned int oPixelPosCol = iPixelPosCol - S; // removing the origin
            unsigned int oPixelPosRow = iPixelPosRow - S;
            unsigned int oPixelPos = oPixelPosRow * W + oPixelPosCol;

            unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;

            d_h[oPixelPos] = 0.0;
            for( int i = -S; i <= S; i++ ) {
                for( int j = -S; j <= S; j++ ) {
                    int tilePixelPosOffset = i * tileW + j;
                    int coefPos = ( i + S ) * kernelSize + ( j + S );
                    d_h[oPixelPos] += sData[ tilePixelPos + tilePixelPosOffset ] * d_cFilterKernel[coefPos];
                }
            }

        }

    }

}