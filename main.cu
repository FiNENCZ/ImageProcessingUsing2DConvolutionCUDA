#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include "pngio.h"

#define BLOCK_SIZE 16

__global__ void blurKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    // Indexy pixelu
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Sdílená paměť pro blok
    __shared__ unsigned char sharedBlock[BLOCK_SIZE + 2][BLOCK_SIZE + 2][3]; // +2 pro okrajovou část

    // Globální index v jednorozměrném poli
    int index = (y * width + x) * 3;

    // Nahrání pixelu do sdílené paměti
    if (x < width && y < height) {
        for (int c = 0; c < 3; c++) {
            sharedBlock[threadIdx.y + 1][threadIdx.x + 1][c] = input[index + c];
        }
    }

    // Nahrání okrajových pixelů do sdílené paměti
    if (threadIdx.x == 0 && x > 0) { // Levý okraj
        for (int c = 0; c < 3; c++) {
            sharedBlock[threadIdx.y + 1][0][c] = input[index - 3 + c];
        }
    }
    if (threadIdx.x == blockDim.x - 1 && x < width - 1) { // Pravý okraj
        for (int c = 0; c < 3; c++) {
            sharedBlock[threadIdx.y + 1][BLOCK_SIZE + 1][c] = input[index + 3 + c];
        }
    }
    if (threadIdx.y == 0 && y > 0) { // Horní okraj
        for (int c = 0; c < 3; c++) {
            sharedBlock[0][threadIdx.x + 1][c] = input[index - width * 3 + c];
        }
    }
    if (threadIdx.y == blockDim.y - 1 && y < height - 1) { // Spodní okraj
        for (int c = 0; c < 3; c++) {
            sharedBlock[BLOCK_SIZE + 1][threadIdx.x + 1][c] = input[index + width * 3 + c];
        }
    }

    __syncthreads();

    // Aplikace průměrovacího filtru (konvoluce)
    if (x < width && y < height) {
        for (int c = 0; c < 3; c++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += sharedBlock[threadIdx.y + 1 + dy][threadIdx.x + 1 + dx][c];
                }
            }
            output[index + c] = sum / 9; // Průměr
        }
    }
}

/**
 * Host function that loads an image, applies the blur effect, and saves the result.
 */
int main() {
    std::string inputFileName = "../input.png";
    std::string outputFileName = "../output.png";

    // Načtení vstupního obrázku
    png::image<png::rgb_pixel> inputImage(inputFileName);
    int width = inputImage.get_width();
    int height = inputImage.get_height();
    size_t imageSize = width * height * 3 * sizeof(unsigned char);

    unsigned char *h_input = new unsigned char[imageSize];
    unsigned char *h_output = new unsigned char[imageSize];

    // Převod obrázku na pole RGB
    pvg::pngToRgb(h_input, inputImage);

    // Alokace GPU paměti
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Kopírování dat na GPU
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Konfigurace kernelu
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Spuštění kernelu
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Kopírování výsledku zpět na CPU
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Uložení výsledného obrázku
    png::image<png::rgb_pixel> outputImage(width, height);
    pvg::rgbToPng(outputImage, h_output);
    outputImage.write(outputFileName);

    // Uvolnění paměti
    delete[] h_input;
    delete[] h_output
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Blurred image saved to " << outputFileName << std::endl;
    return 0;
}