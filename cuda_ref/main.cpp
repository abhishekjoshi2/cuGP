#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common/cycleTimer.h"
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

void check_cholesky(double *M1, double* targetoutput, int d){
        double diff = 0.0, totaldiff = 0.0;
        for(int i = 0; i < d; i++){
                for(int j = 0; j < d ;j++){
                        double tempval = 0.0;
                        for(int k = 0; k < d; k++){
                                tempval += M1[i*d + k] * M1[j * d + k];
                        }
                        diff = tempval - targetoutput[i * d + j];
                        totaldiff += abs(diff);
                }
        }
        printf("FINAL ERROR = %lf\n", totaldiff);
}


void get_symmetric_matrix_1d(double *M, double **matrix1, int dim) {

        srand(time(NULL));
        int setter = 1;
        for (int i = 0; i < dim; i++)
        {
                for (int j = 0; j < dim; j++){
                        matrix1[i][j] = rand() % 100 + 1;
                }
        }
        for (int i = 0; i < dim; i++){
                for(int j = 0; j < dim; j++){
                        M[i * dim + j ] = 0.0;
                        for(int k = 0; k < dim; k++){
                                //M[i * dim + j] += matrix1[i][k]*matrix2[k][j];
                                M[i * dim + j] += matrix1[i][k]*matrix1[j][k];
                        }
                }
        }
}


/********/
/* MAIN */
/********/
int main(){

	double startime, endtime;
	const int Nrows = 7000;
	const int Ncols = Nrows;


	
	int dimp = Nrows;
	double *h_A = new double[dimp*dimp];
	
	double **m1 = new double *[dimp];
	double *temp_m = new double[dimp * dimp];
        for (int i = 0; i < dimp; i++)
        {
                m1[i] = new double[dimp];
        }

	get_symmetric_matrix_1d(temp_m, m1, dimp);

	double *d_A; //device ka banda
        cudacall(cudaMalloc(&d_A, sizeof(double) * dimp * dimp));
        cudacall(cudaMemcpy(d_A, temp_m,  sizeof(double) * dimp * dimp, cudaMemcpyHostToDevice));

	/*
	for(int i = 0 ; i < dimp; i++){
		for(int j = 0 ; j < dimp ;j++){

			printf("%lf ", temp_m[i*dimp + j]);
		}
		printf("\n");
	}	*/

	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;           cudaMalloc(&devInfo,          sizeof(int));

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- CUDA CHOLESKY initialization
	(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, Nrows, d_A, Nrows * Nrows, &work_size));

	// --- CUDA POTRF execution
	double *work;   cudacall(cudaMalloc(&work, work_size * sizeof(double)));
	startime = CycleTimer::currentSeconds();
	(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, Nrows, d_A, Nrows, work, work_size, devInfo));
	endtime = CycleTimer::currentSeconds();
	
	int devInfo_h = 0;  cudacall(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) std::cout   << "Unsuccessful potrf execution\n\n";
	

	printf("Total time taken in cuSOLVER's cholesky = %lf s\n", endtime - startime);

	// --- At this point, the upper triangular part of A contains the elements of L. Showing this.
	cudacall(cudaMemcpy(h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost));

	for(int i = 0; i < Nrows; i++){
		for(int j = 0; j < Ncols; j++){
			if(j > i) h_A[i*dimp + j] = 0.0;	
		}
	}

	cusolverDnDestroy(solver_handle);

	check_cholesky(h_A, temp_m, dimp);
	return 0;

}
