#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include "pngio.h"

#define BLOCK_SIZE 16

#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	} }



__global__ void blurKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels) {
    // Indexy pixelu
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Sdílená paměť
    __shared__ unsigned char sharedMem[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * 3];

    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    int idx = (y * width + x) * channels;
    int sharedIdx = ((sharedY * (BLOCK_SIZE + 2)) + sharedX) * channels;

    // Načtení pixelů do sdílené paměti
    if (x < width && y < height) {
        sharedMem[sharedIdx] = input[idx];
        sharedMem[sharedIdx + 1] = input[idx + 1];
        sharedMem[sharedIdx + 2] = input[idx + 2];

        // Načtení sousedních pixelů na okrajích
        if (threadIdx.x == 0 && x > 0) {
            sharedMem[(sharedY * (BLOCK_SIZE + 2)) * channels] = input[idx - channels];
        }
        if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1) {
            sharedMem[(sharedY * (BLOCK_SIZE + 2) + BLOCK_SIZE + 1) * channels] = input[idx + channels];
        }
        if (threadIdx.y == 0 && y > 0) {
            sharedMem[(sharedX) * channels] = input[idx - width * channels];
        }
        if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1) {
            sharedMem[((BLOCK_SIZE + 1) * (BLOCK_SIZE + 2) + sharedX) * channels] = input[idx + width * channels];
        }
    }
    __syncthreads();

    // Konvoluce s 3x3 kernelem
    if (x < width && y < height) {
        float kernel[3][3] = {
            {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
            {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
            {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}};

        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int smemIdx = ((sharedY + ky) * (BLOCK_SIZE + 2) + (sharedX + kx)) * channels + c;
                    sum += sharedMem[smemIdx] * kernel[ky + 1][kx + 1];
                }
            }
            output[idx + c] = static_cast<unsigned char>(sum);
        }
    }
}

/**
 * Hlavní funkce
 */
int main(int argc, char **argv) {
    std::string inputFileName = "../input.png";
    std::string outputFileName = "../output.png";

    // Načtení obrázku
    png::image<png::rgb_pixel> inputImage(inputFileName);
    int width = inputImage.get_width();
    int height = inputImage.get_height();
    int channels = 3;

    unsigned char *h_input = new unsigned char[width * height * channels];
    unsigned char *h_output = new unsigned char[width * height * channels];

    pvg::pngToRgb(h_input, inputImage);

    // Alokace paměti na GPU
    unsigned char *d_input, *d_output;
    CUDA_CHECK_RETURN(cudaMalloc(&d_input, width * height * channels));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, width * height * channels));

    CUDA_CHECK_RETURN(cudaMemcpy(d_input, h_input, width * height * channels, cudaMemcpyHostToDevice));

    // Konfigurace kernelu
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    double start = omp_get_wtime();
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    double end = omp_get_wtime();

    cudaMemcpy(h_output, d_output, width * height * channels, cudaMemcpyDeviceToHost);

    // Uložení obrázku
    png::image<png::rgb_pixel> outputImage(width, height);
    pvg::rgbToPng(outputImage, h_output);
    outputImage.write(outputFileName);

    std::cout << "Rozostření dokončeno za " << end - start << " sekund." << std::endl;

    // Uvolnění paměti
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));

    return 0;
}
