#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include "utils/pngio.h"

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (3u)
#define TILE_SIZE (BLOCK_SIZE - (FILTER_SIZE - 1))

#define CUDA_CHECK_RETURN(value) {							\
	cudaError_t err = value;									\
	if (err != cudaSuccess) {									\
		fprintf(stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__);	\
		exit(1);												\
	} }

__constant__ int filter[FILTER_SIZE * FILTER_SIZE];
__constant__ float filterMultiplier;

__global__ void processImg(unsigned char *out, unsigned char *in, size_t pitch, unsigned int width, unsigned int height) {
    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    int x_i = x_o - ((FILTER_SIZE - 1) / 2);
    int y_i = y_o - ((FILTER_SIZE - 1) / 2);

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height))
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
    else
        sBuffer[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    int sum = 0;
    if ((threadIdx.x < TILE_SIZE) && (threadIdx.y < TILE_SIZE)) {
        for (int r = 0; r < FILTER_SIZE; ++r){
            for (int c = 0; c < FILTER_SIZE; ++c) {
                sum += filter[r * FILTER_SIZE + c] * sBuffer[threadIdx.y + r][threadIdx.x + c];
            }
        }

        //printf("filterMultiplier: %f", filterMultiplier);
        sum *= filterMultiplier;
        sum = max(0, min(255, sum));

        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;
    }
}

void setFilter(int filterType) {
    int sharpenFilter[FILTER_SIZE * FILTER_SIZE] = {
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0
    };

    float sharpenFilterMultiplier = 1;

    int boxBlurFilter[FILTER_SIZE * FILTER_SIZE] = {
        1,  1,  1,
        1,  1,  1,
        1,  1,  1
    };

    float boxBlurFilterMultiplier = 1.0/(FILTER_SIZE * FILTER_SIZE);

    int gaussianBlurFilter[FILTER_SIZE * FILTER_SIZE] = {
        1,  2,  1,
        2,  4,  2,
        1,  2,  1
    };

    
    float gaussianBlurFilterMultiplier = 1.0/16;

    int edgeDetectionFilter[FILTER_SIZE * FILTER_SIZE] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    float edgeDetectionFilterMultiplier = 1;


    int embossFilter[FILTER_SIZE * FILTER_SIZE] = {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };

    float embossFilterMultiplier = 1;

    
    int scharrHorizontalFilter[FILTER_SIZE * FILTER_SIZE] = {
        3,  0,  -3,
        10, 0,  -10,
        3,  0,  -3
    };

    float scharrHorizontalFilterMultiplier = 1;



    switch (filterType) {
        case 1:
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, sharpenFilter, sizeof(sharpenFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &sharpenFilterMultiplier, sizeof(sharpenFilterMultiplier)));
            break;
        case 2:
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, boxBlurFilter, sizeof(boxBlurFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &boxBlurFilterMultiplier, sizeof(boxBlurFilterMultiplier)));
            break;
        case 3:
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, gaussianBlurFilter, sizeof(gaussianBlurFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &gaussianBlurFilterMultiplier, sizeof(gaussianBlurFilterMultiplier)));
            break;
        case 4:
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, edgeDetectionFilter, sizeof(edgeDetectionFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &edgeDetectionFilterMultiplier, sizeof(edgeDetectionFilterMultiplier)));
            break;
        case 5:
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, embossFilter, sizeof(embossFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &embossFilterMultiplier, sizeof(embossFilterMultiplier)));
            break;
        case 6:
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, scharrHorizontalFilter, sizeof(scharrHorizontalFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &scharrHorizontalFilterMultiplier, sizeof(scharrHorizontalFilterMultiplier)));
            break;
        default:
            fprintf(stderr, "Unknown filter type! Defaulting to Sharpen.\n");
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filter, sharpenFilter, sizeof(sharpenFilter)));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(filterMultiplier, &sharpenFilterMultiplier, sizeof(sharpenFilterMultiplier)));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filter_type (1=Sharpen, 2=Box Blur,3=Gaussian Blur, 4=Edge Detection, 5=Emboss, 6=Scharr (Horizontal)>\n", argv[0]);
        return 1;
    }

    int filterType = atoi(argv[1]);
    setFilter(filterType);

    std::cout << "Loading Image..." << std::endl;

    png::image<png::rgb_pixel> img("../input.png");

    unsigned int width = img.get_width();
    unsigned int height = img.get_height();

    int size = width * height * sizeof(unsigned char);

    unsigned char *h_r = (unsigned char *)malloc(size);
    unsigned char *h_g = (unsigned char *)malloc(size);
    unsigned char *h_b = (unsigned char *)malloc(size);

    unsigned char *h_r_n = (unsigned char *)malloc(size);
    unsigned char *h_g_n = (unsigned char *)malloc(size);
    unsigned char *h_b_n = (unsigned char *)malloc(size);

    pvg::pngToRgb3(h_r, h_g, h_b, img);

    unsigned char *d_r = NULL, *d_g = NULL, *d_b = NULL;
    unsigned char *d_r_n = NULL, *d_g_n = NULL, *d_b_n = NULL;

    size_t pitch_r = 0, pitch_g = 0, pitch_b = 0;

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r, &pitch_r, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, &pitch_g, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b, &pitch_b, width, height));

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice));

    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    double start = omp_get_wtime();
    processImg<<<grid_size, block_size>>>(d_r_n, d_r, pitch_r, width, height);
    processImg<<<grid_size, block_size>>>(d_g_n, d_g, pitch_g, width, height);
    processImg<<<grid_size, block_size>>>(d_b_n, d_b, pitch_b, width, height);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    double end = omp_get_wtime();

    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));

    pvg::rgb3ToPng(img, h_r_n, h_g_n, h_b_n);
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
    CUDA_CHECK_RETURN(cudaFree(d_g_n)); 
    CUDA_CHECK_RETURN(cudaFree(d_b_n));

    return 0;
}
