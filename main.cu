#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include "utils/pngio.h"

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (3u)
#define TILE_SIZE (BLOCK_SIZE-(FILTER_SIZE-1))

#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	} }

__global__ void processImg(unsigned char *out,unsigned char *in, size_t pitch, unsigned int width,unsigned int height){
    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    //due to 2 pixels will be outside
    int x_i = x_o - ((FILTER_SIZE-1)/2);
    int y_i = y_o - ((FILTER_SIZE-1)/2);

    //defining shared memory
    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    //copying into shared memory
    if ((x_i>=0) && (x_i< width) && (y_i>=0) && (y_i< height))
        sBuffer[threadIdx.y][threadIdx.x] = in [y_i * pitch + x_i];
    else //outside of image
        sBuffer[threadIdx.y][threadIdx.x] = 0;


    __syncthreads();

    int sum = 0;
    if((threadIdx.x < TILE_SIZE) && (threadIdx.y < TILE_SIZE)) {
        //applying the filter
        for (int r = 0; r < FILTER_SIZE ;++r)
            for (int c = 0; c < FILTER_SIZE ; ++c)
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c];
    

    sum = sum / (FILTER_SIZE * FILTER_SIZE);
    //write into the output    
    if ( x_o < width && y_o <height)
        out[y_o * width + x_o] = sum;
    }

}

 
int main(int argc, char **argv) {
    std::cout << "Loading Image..." << std::endl;

    //Loading funcion
    png::image<png::rgb_pixel> img("../input.png");

    unsigned int width = img.get_width();
    unsigned int height = img.get_height();

    //Defining size to allocate memory
    int size = width * height * sizeof(unsigned char);

    //Allocating memory buffer to host memory
    unsigned char *h_r = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_g = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_b = (unsigned char*) malloc (size * sizeof(unsigned char));

    //Allocating memory for the output
    unsigned char *h_r_n = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_g_n = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_b_n = (unsigned char*) malloc (size * sizeof(unsigned char));


    //convert image to raw buffer ( de-channel the iamge colors )
    pvg::pngToRgb3(h_r,h_g,h_b,img);

    //memory allocation of the device ( GPU )

    //allocate output memory
    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n,size));


    //allocate input memory of the device
    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    //defining the pitch size
    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;


    //allocate memory with pitching
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r,&pitch_r,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g,&pitch_g,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b,&pitch_b,width,height));


    //copying the memory to device
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice));

    //configuring image kernel
    dim3 grid_size( (width + TILE_SIZE -1)/TILE_SIZE,
    (height + TILE_SIZE -1)/TILE_SIZE);

    dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);


    //defining the kernel function:
    double start = omp_get_wtime();
    processImg<<<grid_size,blockSize>>>(d_r_n, d_r ,pitch_r ,width,height);
    processImg<<<grid_size,blockSize>>>(d_g_n, d_g ,pitch_g ,width,height);
    processImg<<<grid_size,blockSize>>>(d_b_n, d_b ,pitch_b ,width,height);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    double end = omp_get_wtime();


    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n,d_r_n,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n,d_g_n,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n,d_b_n,size,cudaMemcpyDeviceToHost));

    pvg::rgb3ToPng(img, h_r_n, h_g_n, h_b_n); //combining the channels into one image in png format
    
    std::cout << "Done in " << end - start << " seconds." << std::endl;

    img.write("../output.png");


    free(h_r);
    free(h_g);
    free(h_b);

    free(h_r_n);
    free(h_g_n);
    free(h_b_n);

    CUDA_CHECK_RETURN(cudaFree(d_r));
    CUDA_CHECK_RETURN(cudaFree(d_g));
    CUDA_CHECK_RETURN(cudaFree(d_b));

    CUDA_CHECK_RETURN(cudaFree(d_r_n));
    CUDA_CHECK_RETURN(cudaFree(d_b_n));
    CUDA_CHECK_RETURN(cudaFree(d_g_n));

    return 0;
}