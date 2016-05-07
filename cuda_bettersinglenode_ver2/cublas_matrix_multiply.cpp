#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/cycleTimer.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void caller();

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const double alf = 1;
     const double bet = 0;
     const double *alpha = &alf;
     const double *beta = &bet;
 
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
 
     // Do the actual multiplication
     cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 
     // Destroy the handle
     cublasDestroy(handle);
 }


int main(){
	caller();
	return 0;
			
	int m = 10;
	int n = m;
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context
	int i,j; // i-row index , j-col. index
	double * a; // mxm matrix a on the host
	double * b; // mxm matrix b on the host
	double * c; // mxm matrix c on the host
	a=( double *) malloc (m*m* sizeof ( double )); // host memory for a

	b=( double *) malloc (m*m* sizeof ( double )); // host memory for b
	c=( double *) malloc (m*m* sizeof ( double )); // host memory for b

	int ind =1; // a:
	for(j=0;j<m;j ++){ // 11
		for(i=0;i<m;i ++){ // 12 ,17
	//		if(i >=j){ // 13 ,18 ,22
				a[ IDX2C(i,j,m)]=( float )ind ++; // 14 ,19 ,23 ,26
	//		} // 15 ,20 ,24 ,27 ,29
		} // 16 ,21 ,25 ,28 ,30 ,31
	}
	 printf (" lower triangle of a:\n");
	for (i=0;i<m;i ++){
		for (j=0;j<m;j ++){
	//		if(i >=j)
				printf (" %5.0f",a[ IDX2C(i,j,m)]);
		}
		printf ("\n");
	} 

	ind =11; // b:
	for(j=0;j<n;j ++){ // 11 ,17 ,23 ,29 ,35
		for(i=0;i<m;i ++){ // 12 ,18 ,24 ,30 ,36
			if(i == j)
			b[ IDX2C(i,j,m)] = 1.0; // 13 ,19 ,25 ,31 ,37
			else 
			b[ IDX2C(i, j, m)] = 2.0;
			//ind ++; // 14 ,20 ,26 ,32 ,38
		} // 15 ,21 ,27 ,33 ,39
	} // 16 ,22 ,28 ,34 ,40
	 printf ("b:\n");
	for (i=0;i<m;i ++){
		for (j=0;j<n;j ++){
			printf (" %5.0f",b[IDX2C(i,j,m)]); // print b row by row
		}
		printf ("\n");
	} 

	double * d_a; // d_a - a on the device
	double * d_b; // d_b - b on the device
	double * d_c; // d_c - c on the devicde
	cudaStat = cudaMalloc (( void **)& d_a ,m*m* sizeof (*a)); // device memory alloc for a
	cudaStat = cudaMalloc (( void **)& d_b ,m*m* sizeof (*b)); // device memory alloc for b
	cudaStat = cudaMalloc (( void **)& d_c ,m*m* sizeof (*c)); // device memory alloc for c

	stat = cublasCreate (& handle ); // initialize CUBLAS context

	stat = cublasSetMatrix (m,m, sizeof (*a) ,a,m,d_a ,m); //a -> d_a
	stat = cublasSetMatrix (m,m, sizeof (*b) ,b,m,d_b ,m); //b -> d_b

	double startime = CycleTimer::currentSeconds();
	gpu_blas_mmul(d_a, d_b, d_c, m, m, m);
	double endtime = CycleTimer::currentSeconds();
	
	stat = cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); // d_b -> b
	 printf (" solution x from Strsm :\n");
	for(i=0;i<m;i ++){
		for(j=0;j<n;j ++){
			printf (" %11.5f",c[IDX2C(i,j,m )]); // print b after Strsm
		}
		printf ("\n");
	} 
	cudaFree (d_a ); // free device memory
	cudaFree (d_b ); // free device memory
	cudaFree (d_c ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	free (a); // free host memory
	free (b); // free host memory
	free (c); // free host memory

	printf("Time taken: %lf\n", endtime - startime);
	return EXIT_SUCCESS ;
}
