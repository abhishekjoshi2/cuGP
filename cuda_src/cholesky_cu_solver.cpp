#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"

#define cudacall(call) \
{ \
	cudaError_t err = (call);                                                                                               \
	if(cudaSuccess != err)                                                                                                  \
	{                                                                                                                       \
		fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
		cudaDeviceReset();                                                                                                  \
		exit(EXIT_FAILURE);                                                                                                 \
	}                                                                                                                       \
} \
/********/
/* MAIN */
/********/
int main(){

	const int Nrows = 5;
	const int Ncols = 5;

	// --- Setting the host, Nrows x Ncols matrix
		double h_A[Nrows][Ncols] = { 
			{ 1.,    -1.,    -1.,    -1.,    -1.,},  
			{-1.,     2.,     0.,     0.,     0.,}, 
			{-1.,     0.,     3.,     1.,     1.,}, 
			{-1.,     0.,     1.,     4.,     2.,}, 
			{-1.,     0.,     1.,     2.,     5.,}
		};

	printf("Original matrix\n");
	for(int i = 0; i < Nrows; i++)
	{
		for(int j = 0; j < Ncols; j++)
			printf("%f\t", h_A[i][j]);
		printf("\n");
	}

	// --- Setting the device matrix and moving the host matrix to the device
	double *d_A;
	cudacall(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
	cudacall(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;
	cudacall(cudaMalloc(&devInfo, sizeof(int)));

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- CUDA CHOLESKY initialization
	cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, &work_size);

	// --- CUDA POTRF execution
	double *work;
	cudacall(cudaMalloc(&work, work_size * sizeof(double)));
	cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, work, work_size, devInfo);
	int devInfo_h = 0;
	cudacall(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) std::cout   << "Unsuccessful potrf execution\n\n";

	// --- At this point, the upper triangular part of A contains the elements of L. Showing this.
	printf("\nFactorized matrix\n");
	cudacall(cudaMemcpy(h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost));
	for(int i = 0; i < Nrows; i++)
	{
		for(int j = 0; j < Ncols; j++)
		{
			printf("%f\t", h_A[i][j]);
		}
		printf("\n");
	}	

	cusolverDnDestroy(solver_handle);

	return 0;
}
