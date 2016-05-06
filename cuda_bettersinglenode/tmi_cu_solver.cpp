#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/cycleTimer.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 4096 // a - mxm matrix
#define n 4096 // b,x - mxn matrices

int main ( void ){
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context
	int i,j; // i-row index , j-col. index
	double * a; // mxm matrix a on the host
	double * b; // mxn matrix b on the host
	a=( double *) malloc (m*m* sizeof ( double )); // host memory for a

	b=( double *) malloc (m*n* sizeof ( double )); // host memory for b

	int ind =11; // a:
	for(j=0;j<m;j ++){ // 11
		for(i=0;i<m;i ++){ // 12 ,17
			if(i >=j){ // 13 ,18 ,22
				a[ IDX2C(i,j,m)]=( double )ind ++; // 14 ,19 ,23 ,26
			} // 15 ,20 ,24 ,27 ,29
		} // 16 ,21 ,25 ,28 ,30 ,31
	}
	/* printf (" lower triangle of a:\n");
	for (i=0;i<m;i ++){
		for (j=0;j<m;j ++){
			if(i >=j)
				printf (" %5.0f",a[ IDX2C(i,j,m)]);
		}
		printf ("\n");
	} */

	ind =11; // b:
	for(j=0;j<n;j ++){ // 11 ,17 ,23 ,29 ,35
		for(i=0;i<m;i ++){ // 12 ,18 ,24 ,30 ,36
			b[ IDX2C(i,j,m)] = ind++;
			/*if(i == j)
			b[ IDX2C(i,j,m)] = 1.0; // 13 ,19 ,25 ,31 ,37
			else 
			b[ IDX2C(i, j, m)] = 0.0;*/
			//ind ++; // 14 ,20 ,26 ,32 ,38
		} // 15 ,21 ,27 ,33 ,39
	} // 16 ,22 ,28 ,34 ,40
	/* printf ("b:\n");
	for (i=0;i<m;i ++){
		for (j=0;j<n;j ++){
			printf (" %5.0f",b[IDX2C(i,j,m)]); // print b row by row
		}
		printf ("\n");
	} */

	double * d_a; // d_a - a on the device
	double * d_b; // d_b - b on the device
	cudaStat = cudaMalloc (( void **)& d_a ,m*m* sizeof (*a)); // device
	// memory alloc for a
	cudaStat = cudaMalloc (( void **)& d_b ,m*n* sizeof (*b)); // device
	// // memory alloc for b
	stat = cublasCreate (& handle ); // initialize CUBLAS context

	stat = cublasSetMatrix (m,m, sizeof (*a) ,a,m,d_a ,m); //a -> d_a
	stat = cublasSetMatrix (m,n, sizeof (*b) ,b,m,d_b ,m); //b -> d_b
	double al =1.0f;

	double startime = CycleTimer::currentSeconds();
	(cublasDtrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,m,n,&al,d_a,m,d_b,m));
	stat = cublasGetMatrix (m,n, sizeof (*b) ,d_b ,m,b,m); // d_b -> b
	double endtime = CycleTimer::currentSeconds();
	/* printf (" solution x from Strsm :\n");
	for(i=0;i<m;i ++){
		for(j=0;j<n;j ++){
			printf (" %11.5f",b[IDX2C(i,j,m )]); // print b after Strsm
		}
		printf ("\n");
	} */
	cudaFree (d_a ); // free device memory
	cudaFree (d_b ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	free (a); // free host memory
	free (b); // free host memory

	printf("Time taken: %lf\n", endtime - startime);
	return EXIT_SUCCESS ;
}
