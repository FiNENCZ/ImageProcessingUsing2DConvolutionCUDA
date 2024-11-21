/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>

#include "pngio.h"

#define WIDTH (800u)
#define HEIGHT (600u)
#define MAX_ITER (7650u)

#define BLOCK_SIZE (16u)

#define USE_GPU 1

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	} }


// Gaussovské jádro pro rozostření 3x3
__constant__ float d_kernel[3][3] = {
    { 0.0751f, 0.1238f, 0.0751f },
    { 0.1238f, 0.2042f, 0.1238f },
    { 0.0751f, 0.1238f, 0.0751f }
};

// CUDA kernel pro 2D konvoluci
__global__ void applyGaussianBlur(unsigned char* d_img, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float r = 0.0f, g = 0.0f, b = 0.0f;

        // Procházení jádrem 3x3
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int x_offset = min(max(x + j, 0), width - 1);
                int y_offset = min(max(y + i, 0), height - 1);
                int index = (y_offset * width + x_offset) * 3;

                r += d_img[index] * d_kernel[i + 1][j + 1];
                g += d_img[index + 1] * d_kernel[i + 1][j + 1];
                b += d_img[index + 2] * d_kernel[i + 1][j + 1];
            }
        }

        int index = (y * width + x) * 3;
        d_output[index] = static_cast<unsigned char>(r);
        d_output[index + 1] = static_cast<unsigned char>(g);
        d_output[index + 2] = static_cast<unsigned char>(b);
    }
}

void processImage(const std::string& inputFileName, const std::string& outputFileName) {
    // Načtení obrázku pomocí pvg knihovny
    png_img_t inputImage;
    inputImage.read(inputFileName);

    int width = inputImage.get_width();
    int height = inputImage.get_height();

    // Alokace hostitelské paměti pro obrázek a výstupní obrázek
    unsigned char* h_img = new unsigned char[width * height * 3];
    unsigned char* h_output = new unsigned char[width * height * 3];

    // Načtení RGB dat do h_img
    pvg::pngToRgb(h_img, inputImage);

    // Alokace paměti na zařízení (GPU)
    unsigned char* d_img;
    unsigned char* d_output;
    CUDA_CHECK_RETURN(cudaMalloc(&d_img, width * height * 3 * sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));

    // Kopírování dat z hostitele na zařízení
    CUDA_CHECK_RETURN(cudaMemcpy(d_img, h_img, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Nastavení velikosti bloků a mřížky pro CUDA
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Spuštění CUDA kernelu pro aplikaci Gaussovského rozostření
    applyGaussianBlur<<<gridSize, blockSize>>>(d_img, d_output, width, height);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Kontrola chyby
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Kopírování výsledků zpět z GPU na hostitele
    CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Uložení výsledného obrázku pomocí pvg knihovny
    png_img_t outputImage(width, height);
    pvg::rgbToPng(outputImage, h_output);
    outputImage.write(outputFileName);

    // Uvolnění alokované paměti
    delete[] h_img;
    delete[] h_output;
    CUDA_CHECK_RETURN(cudaFree(d_img));
    CUDA_CHECK_RETURN(cudaFree(d_output));
}

int main(int argc, char** argv) {

    std::string inputFileName = "../input.png";
    std::string outputFileName = "../output.png";

    processImage(inputFileName, outputFileName);

    return 0;
}