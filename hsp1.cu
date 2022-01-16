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






void MatrixInit(float* M, int n, int p) {
	float LO = -1.;
	float HI = 1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			float rd = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
			M[n * i + j] = rd;
		}
	}
}
void MatrixInit_value(float* M, int n, int p,float value) {

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			M[n * i + j] = value;
		}
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

int main() {
	
	/*  
	 //Allocate memory for our matrices
	float * a, * b, * out;
	float* d_a, * d_b, * d_out;

	// Allocate memory
	a = (float*)malloc(sizeof(float) * N*N);
	b = (float*)malloc(sizeof(float) * N*N);
	out = (float*)malloc(sizeof(float) * N*N);
	MatrixInit(a, N,N);
	MatrixInit(b,N,N);
	printf("matrix 1");
	//MatrixPrint(a, 2,2); // just print first 4 numbers
	//printf("a[0] = %f\n", a[0]);

	// ===>  Matrix add cpu

	MatrixAdd(a, b, out, N, N);

	// ===> Matrix add GPU

	cudaMalloc((void**)&d_a, sizeof(float) * N*N);
	cudaMalloc((void**)&d_b, sizeof(float) * N*N);
	cudaMalloc((void**)&d_out, sizeof(float) * N*N);

	cudaMemcpy(d_a, a, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
	cudaMatrixAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_out, N,N);

	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N*N; i++) {
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
	printf("PASSED\n");
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	*/

	/*
	float* a1, * b1, * out1, * out2;
	float* d_a1, * d_b1, * d_out1, * d_out2;


	clock_t t;
	t = clock();

	// ===>  Matrix MULT CPU

	// Allocate memory
	a1 = (float*)malloc(sizeof(float) * N * N);
	b1 = (float*)malloc(sizeof(float) * N * N);
	out1 = (float*)malloc(sizeof(float) * N * N);
	out2 = (float*)malloc(sizeof(float) * N * N);
	//MatrixInit(a1, N, N);
	//MatrixInit(b1, N, N);
	MatrixInit_value(a1, N, N, 1.);
	MatrixInit_value(b1, N, N, 1.);
	//b1[3] = 0.;

	MatrixInit_value(out1, N, N, 0.);
	MatrixInit_value(out2, N, N, 0.);

	MatrixMult(a1, b1, out1,N);

	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	printf("mult took %f seconds to execute \n", time_taken);

	t = clock();

	// ===>   Matrix MULT GPU

	cudaMalloc((void**)&d_a1, sizeof(float) * N * N);
	cudaMalloc((void**)&d_b1, sizeof(float) * N * N);
	//cudaMalloc((void**)&d_out1, sizeof(float) * N * N);
	cudaMalloc((void**)&d_out2, sizeof(float) * N * N);

	cudaMemcpy(d_a1, a1, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b1, b1, sizeof(float) * N*N, cudaMemcpyHostToDevice);
	//int threads = 32;
	//int blocks = (N + threads - 1) / threads;
	int THREADS = 32;

	// Blocks per grid dimension (assumes THREADS divides N evenly)
	int BLOCKS = N / THREADS;
	
	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Launch kernel
	cudaMatrixMult<<<blocks, threads >>>(d_a1, d_b1, d_out2, N);

	t = clock()-t;
	time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	printf("mult cuda took %f seconds to execute \n", time_taken);

	cudaMemcpy(out2, d_out2, sizeof(float) * N*N, cudaMemcpyDeviceToHost);
	//test
	float error = 0.;
	/*
	printf("res1\n");
	MatrixPrint(a1, 2, 2); // just print first 4 numbers
	printf("res2\n");
	MatrixPrint(b1, 2, 2); // just print first 4 numbers


	printf("res1\n");
	MatrixPrint(out1, 2, 2); // just print first 4 numbers
	printf("res2\n");
	MatrixPrint(out2, 2, 2); // just print first 4 numbers
	
	for (int i = 0; i < N * N; i++) {
		error = out1[i] - out2[i];
		//printf("error %f\n", error);
		//assert(fabs(error) < MAX_ERR);
	}
	//printf("error %f\n", error);
	printf("MULT PASSED\n");


	cudaFree(d_a1);
	cudaFree(d_b1);
	cudaFree(d_out2);
	*/


float* a2, * kernel, * out3, * out4;
float* d_a2, * d_kernel, * d_out3, * d_out4;


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
}
