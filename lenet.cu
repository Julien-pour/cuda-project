#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "device_launch_parameters.h"

#include <stdio.h>

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <time.h>

#define N 1024//3200 test with multiplication
#define MAX_ERR 1e-6






void MatrixInit(float* M, int n) {
	float LO = 0.;
	float HI = 1;
	for (int i = 0; i < n; i++) {
			float rd = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
			M[i] = rd;
		
	}
}
void MatrixInit_value(float* M, int n,float value) {

	for (int i = 0; i < n; i++) {
			M[i] = value;
		
	}
}

void MatrixPrint(float* M, int n, int p) {
	printf("matrix");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			float val = M[n * i + j];
			printf("M[%d][%d] %f  ", i,j,val);
		}
		printf("\n");
	}
}

void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
	for (int i = 0; i < n*p; i++) {
		Mout[i] = M1[i] + M2[i];
	}
}

__global__ void cudaMatrixAdd(float* a, float* b, float* out, int n, int p) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n * p) {
		out[i] = a[i] + b[i];
	}
}

void MatrixMult(float* M1, float* M2, float* Mout, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float res = 0.0;
			for (int k = 0; k < n; k++) {
				res += M1[i*n + k] * M2[k*n + j];
			}
			Mout[i*n + j] = res;
		}
	}
}



__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float Sum = 0;

	if (row < n && col < n) {
		for (int i = 0; i < n; i++) {
			Sum += M1[row * n + i] * M2[i * n + col];
		}

	}
	Mout[row * n + col] = Sum;
}







__global__ void convolution_kernel(float* output, float* input, float* filter,int size_image,int size_kernel) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i, j;
	float sum = 0.0;
	int filter_width = size_kernel;
	int filter_height = size_kernel;
	int image_width = size_image;
	int image_height = size_image;
	if (y < image_height && x < image_width) {

		for (j = 0; j < filter_height; j++) {
			for (i = 0; i < filter_width; i++) {
				sum += input[(y + j) * image_width + (x + i)] * filter[j * filter_width + i];
			}
		}

		output[y * image_width + x] = sum;
	}
}

__global__ void activation_tanh(float* M, float n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n ) {
		float res = tanhf(M[1]);
		M[i] = res;
	}
}

int main() {

// ===>   LeNet-5
float* raw_data, * C1_data, * S1_data, * C1_kernel;
float* d_raw_data, * d_C1_data, * d_S1_data, * d_C1_kernel;

int size_img = 32;
raw_data = (float*)malloc(sizeof(float) * size_img * size_img); //image
MatrixInit(raw_data, size_img * size_img);
printf("raw img \n");
MatrixPrint(raw_data, 10, 10); // just print first 10 numbers
printf("\n");


int size_c1 = 28;
int size_channel_c1 = 1;
C1_data = (float*)malloc(sizeof(float) * size_c1 * size_c1 * size_channel_c1); //1st conv
MatrixInit_value(C1_data, size_c1 * size_c1 * size_channel_c1, 0.);


int size_S1 = 14;
S1_data = (float*)malloc(sizeof(float) * size_S1 * size_S1 * size_channel_c1); //1st average pool
MatrixInit_value(S1_data, size_S1 * size_S1 * size_channel_c1, 0.);


int size_c1_k = 5;
C1_kernel = (float*)malloc(sizeof(float) * size_channel_c1 * size_c1_k * size_c1_k); //1st conv
MatrixInit(C1_kernel, size_channel_c1 * size_c1_k * size_c1_k);
printf("kernel \n");
MatrixPrint(C1_kernel, 5, 5); // just print first 4 numbers
printf("\n");


float* C1_kernel_inter= (float*)malloc(sizeof(float)  * size_c1_k * size_c1_k); 
for (int j = 0; j < size_c1_k * size_c1_k; j++) {
	C1_kernel_inter[j] = C1_kernel[j];
}



// ====> 1st conv 

cudaMalloc((void**)&d_raw_data, sizeof(float) * size_img * size_img);
cudaMalloc((void**)&d_C1_kernel, sizeof(float) * size_c1_k * size_c1_k);
cudaMalloc((void**)&d_C1_data, sizeof(float) * size_c1 * size_c1);

cudaMemcpy(d_raw_data, raw_data, sizeof(float) * size_img * size_img, cudaMemcpyHostToDevice);
cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * size_c1_k * size_c1_k, cudaMemcpyHostToDevice);


int THREADS = 32;

// Blocks per grid dimension (assumes THREADS divides N evenly)
int BLOCKS = ceil(size_img / THREADS);


// Use dim3 structs for block  and grid dimensions
dim3 threads(THREADS, THREADS);
dim3 blocks(BLOCKS, BLOCKS);

// Launch kernel
convolution_kernel<<<blocks, threads>>>(d_C1_data, d_raw_data, d_C1_kernel, size_img, 5);

// ===> tanh 

int NUM_THREADS = 1024;
int NUM_BLOCKS = (size_c1 * size_c1 + NUM_THREADS - 1) / NUM_THREADS;
activation_tanh<<<NUM_BLOCKS,NUM_THREADS>>>(d_C1_data, size_c1 * size_c1);



cudaMemcpy(C1_data, d_C1_data, sizeof(float) * size_c1* size_c1, cudaMemcpyDeviceToHost);

printf("1st conv \n");

MatrixPrint(C1_data, 10, 10); // just print first 4 numbers



//  ====> 1st Average pooling 

float* S1_kernel;
float* d_S1_kernel;



int size_S1_kernel = 2;
S1_kernel = (float*)malloc(sizeof(float)  * size_S1_kernel * size_S1_kernel); //1st avg

MatrixInit_value(S1_kernel, size_S1_kernel * size_S1_kernel, 0.25); //init avg kernel


cudaMalloc((void**)&d_S1_kernel, sizeof(float) * size_S1_kernel * size_S1_kernel);
cudaMalloc((void**)&d_S1_data, sizeof(float) * size_c1 * size_c1);

cudaMemcpy(d_C1_data, C1_data, sizeof(float) * size_c1 * size_c1, cudaMemcpyHostToDevice); //est-ce nécessaire ?
cudaMemcpy(d_S1_kernel, S1_kernel, sizeof(float) * size_S1_kernel * size_S1_kernel, cudaMemcpyHostToDevice);


THREADS = 28; 

// Blocks per grid dimension (assumes THREADS divides N evenly)
BLOCKS = ceil(size_c1 / THREADS);


// Use dim3 structs for block  and grid dimensions
dim3 threads1(THREADS, THREADS);
dim3 blocks1(BLOCKS, BLOCKS);

// Launch kernel
convolution_kernel <<<blocks1, threads1 >>>(d_S1_data, d_C1_data, d_S1_kernel, size_S1, 2);

cudaMemcpy(S1_data, d_S1_data, sizeof(float) * size_S1 * size_S1, cudaMemcpyDeviceToHost);

printf("1st avg pool \n");

MatrixPrint(S1_data, 5, 5); // just print first 4 numbers

cudaFree(d_C1_data);
cudaFree(d_raw_data);
cudaFree(d_C1_kernel);



/*
float* a2, * kernel, * out3, * out4;
float* d_a2, * d_kernel, * d_out3, * d_out4;
MatrixInit_value(a2, N, N, 1.);


//clock_t t;
//t = clock();


// Allocate memory
a2 = (float*)malloc(sizeof(float) * N * N);
int kernel_size = 7;
kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size);
out3 = (float*)malloc(sizeof(float) * N * N);
out4 = (float*)malloc(sizeof(float) * N * N);
//MatrixInit(a1, N, N);
//MatrixInit(b1, N, N);
MatrixInit_value(a2, N, N, 1.);
MatrixInit_value(kernel, kernel_size, kernel_size, 1.);
//b1[3] = 0.;



cudaMalloc((void**)&d_a2, sizeof(float)* N* N);
cudaMalloc((void**)&d_kernel, sizeof(float)* kernel_size* kernel_size);
//cudaMalloc((void**)&d_out1, sizeof(float) * N * N);
cudaMalloc((void**)&d_out4, sizeof(float)* N* N);

cudaMemcpy(d_a2, a2, sizeof(float)* N* N, cudaMemcpyHostToDevice);
cudaMemcpy(d_kernel, kernel, sizeof(float)* kernel_size* kernel_size, cudaMemcpyHostToDevice);
//int THREADS = 16;
//int BLOCKS = (N + THREADS - 1) / THREADS;
int THREADS = 32;

// Blocks per grid dimension (assumes THREADS divides N evenly)
int BLOCKS = ceil(N / THREADS);


// Use dim3 structs for block  and grid dimensions
dim3 threads(THREADS, THREADS);
dim3 blocks(BLOCKS, BLOCKS);

// Launch kernel
//conv2d<< <blocks, threads >> > (d_a2, d_kernel, d_out4, N);
convolution_kernel<<<blocks, threads>>>(d_out4, d_a2, d_kernel, N,7);
cudaMemcpy(out4, d_out4, sizeof(float)* N* N, cudaMemcpyDeviceToHost);

MatrixPrint(out4, 20, 20); // just print first 4 numbers
MatrixPrint(kernel, 7, 7); // just print first 4 numbers

cudaFree(d_a2);
cudaFree(d_kernel);
cudaFree(d_out4);
*/
}
