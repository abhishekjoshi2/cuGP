#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_functions.h>
#include <cstdlib>
#include "../common/cycleTimer.h"
#include <fstream>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include "../cuda_src/Eigen/Dense"
#include "../cuda_src/Eigen/src/Core/util/DisableStupidWarnings.h"
#include <cmath>
#include<string>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"

#define BLOCK_SIZE 32

//For cublas
int *devInfo;
cusolverDnHandle_t solver_handle;
cublasHandle_t blas_handle;
cublasStatus_t stat;

double *lowermat_inv_store;

bool correct = true;
double *temp_m; 

double *mt;
double *mt_transpose;

double *orig_sym;

// HOST gloval variables
double *lh_host; //Remember to free during cleanup


//For gradient
double *Ksqdist; // for squarred distances
double *matforell; //for gradient of ell
double *tempfsforkinv; //matrix for storing forward substitution result
double *Kinv; //inverse of K matrix
double *Kintact; //want to keep it
double *temp1dvec; //for storing Kinv*y
double *tempWmatrix; // (hard to explain, basically a intermediate step)
double *tempdiagonal; //for storing diagonal elements of tempWmatrix: NOTE: CAN WE REUSE temp1dvec INSTEAD??? CHECK PLEASE

// K is the covariance matrix (will be updated depending upon hyper params
double *K;
double *a11;
double *a21_transpose;
double *l21_transpose_from_fs;
double *l21;
double *l22_temp;  //This is for updating a22


double *X; // training set
double *labels; // labels of the training set (actually regression values)
double *temp_fs; // for saving the result of forward substitution while performing compute_likelihood!!
double *temp_bs; // for saving the result of backward substitution in compute_likelihood
double *ll_dotprod; // for saving the result of the dot product in compute_likelihood
 
double *loghyper;
double *log_det; // log of determinant
double *identity; // for gradient of hp

// N is the number of training samples, and DIM is the number of parameters
int N, DIM;
int totalN; //total number of samples in the dataset; totalN = N + Ntest;


int Ntest; //Ntest is the number of test samples
double *Xtest;
double *labelstest;
double *tmeanvec;
double *tvarvec;
double *Ktest_vec;

// For actual TMI
double *tmi_intermediate_output;

// For testing TMI
 double *lower_triangular_mat;
double *tmi_playground; 

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

double *get_loghyperparam();


__global__ void generate_identity(double *M, int size){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	
	//if (i_index >= size * size)
	//		return;
//	int i = i_index / size;
//	int j = i_index % size;
//	if(i == j) M[i*size + i] = 1.0;
//	else M[i*size + j] = 0.0;	

	if(i_index >= size || j_index >= size) return;
	int mainpoint = j_index * size + i_index;
	if(i_index == j_index) M[mainpoint] = 1.0;
	else M[mainpoint] = 0.0;
}
__global__ void print_vector(double *input, int size){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
        if(i_index >= 1) return;
	for(int i = 0; i < size; i++){
		printf("%lf ", input[i]);
	}
	printf("\n");
}


__global__ void compute_NLPP(double *actualtestlabels, double * predicted_testmean, double *predicted_testvar, int Ntest, double * ans_nlpp){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
        if(i_index >= 1) return;
	double ans = 0.0;
        for(int i = 0; i < Ntest; i++) {
                double val = 0.5 * log(6.283185 * predicted_testvar[i]) + pow( (predicted_testmean[i] - actualtestlabels[i]) , 2) / (2 * predicted_testvar[i]);
		printf("predvar = %lf, predmean = %lf, actualmean = %lf, lpp = %lf\n", predicted_testvar[i], predicted_testmean[i], actualtestlabels[i], val);
                ans += val;
        }
	printf("TO FINAL ANSWER YEH HONA CHAHHIYE: %lf\n", ans / Ntest);
        *ans_nlpp = (ans / Ntest);

}

__global__ void copy_Kmatrix(double *input, double *output, int size){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	
	/*(if (i_index >= size * size)
		return;
	output[i_index] = input[i_index];*/
	if(i_index >= size || j_index >= size) return;
	output[j_index * size + i_index] = input[j_index * size + i_index];

}

__global__ void gather_diagonal(double *inputMat, double *diagvec, int size){
	
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= size)
		return;
	diagvec[i_index] = inputMat[i_index * size + i_index];

}

__global__ void set_upper_zero(double *M, int dim){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

/*	if (i_index >= (dim * dim))
		return;

	int rowN = i_index / dim;
	int colN = i_index % dim;

	if(rowN >= colN) return;

	M[rowN * dim + colN] = 0.0;*/


	if(i_index >= dim || j_index >= dim || j_index >= i_index) return;
	M[j_index*dim + i_index] = 0.0;	
}

	__global__ void
get_determinant_from_L(double *M, int dim, double *log_det)
{
	// single thread

	double ans = 0.0;
	for (int i = 0; i < dim; i++){
		double val = log(M[i * dim + i]);
		ans += val;
	}
	ans *= 2;
	*log_det = ans;
	printf("Determinant is %lf\n", ans);
}

	__global__ void
elementwise_matrix_mult(double *mat1, double *mat2, double *mat3, int rows, int cols)
{
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	
	/*
	int target_row, target_col;
	double dot_product = 0.0;

	target_row = i_index / cols;
	target_col = i_index % cols;

	if (target_row >= rows || target_col >= cols)
		return;

	mat3[target_row * cols + target_col] = mat1[target_row * cols + target_col] * mat2[target_row * cols + target_col];
	*/

	if(i_index >= cols || j_index >= rows) return;
	mat3[j_index * rows + i_index ] = mat1[j_index * rows + i_index] * mat2[j_index * rows + i_index];
}

__global__ void
compute_K_train(double *M, double *K_output, double *loghyper, int n, int dim) {
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

/*	if (i_index >= n * n) return;

	double ell_sq = exp(loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(loghyper[1] * 2); // signal variance
	double noise_var = exp(loghyper[2] * 2); //noise variance

	int M_row, M_col;

	M_row = i_index / n;
	M_col = i_index % n;

	if (M_row < M_col) // upper triangular bye bye
		return;

	if (M_row == M_col){
		K_output[M_row * n + M_col] = signal_var +  noise_var;
		return;
	}

	for (int i = 0; i < dim; i++)
		dot_product += (M[M_row * dim + i] - M[M_col * dim + i]) * (M[M_row * dim + i] - M[M_col * dim + i]);

	dot_product = signal_var * exp(-dot_product * 0.5 / ell_sq);

	K_output[M_row * n + M_col] = K_output[M_col * n + M_row] = dot_product;
*/
	if (i_index >= n || j_index >= n || i_index > j_index) return;

	double ell_sq = exp(loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(loghyper[1] * 2); // signal variance
	double noise_var = exp(loghyper[2] * 2); //noise variance

	double dot_product = 0.0;
	if(i_index == j_index){
		K_output[j_index * n + i_index] = signal_var + noise_var;
		return;
	}
	for(int i = 0 ; i < dim;i++){
		double val1 = M[i_index * dim + i];
		double val2 = M[j_index * dim + i];
		dot_product += (val1 - val2) * (val1 - val2);
	}
	dot_product = signal_var * exp(-dot_product * 0.5 / ell_sq);
	K_output[j_index * n + i_index] = K_output[i_index * n + j_index] = dot_product;
}

__global__ void
compute_K_test(double *M, double *testsample, double *ktest_output_vector, double *loghyper, int n, int dim) {
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= n ) return;

	double ell_sq = exp(loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(loghyper[1] * 2); // signal variance
	double noise_var = exp(loghyper[2] * 2); //noise variance

	double dot_product = 0.0;


	for (int i = 0; i < dim; i++){
		double val1 = M[i_index * dim + i];
		double val2 = testsample[i];
		dot_product += (val1 - val2) * (val1 - val2);
	}

	dot_product = signal_var * exp(-dot_product * 0.5 / ell_sq);

	ktest_output_vector[i_index] = dot_product;
}

__global__ void matrix_vector_multiply(double *M, double *x, double *output, int size) {
	
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= size) return;
	
	double ans = 0.0;
	for(int i = 0; i < size; i++){
		ans += M[i_index * size + i] * x[i];
	}
	output[i_index] = ans;
}

__global__ void vector_matrix_multiply(double *x, double *M, double *output, int size) {
	
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= size) return;
	
	double ans = 0.0;
	for(int i = 0; i < size; i++){
		ans += M[i * size + i_index] * x[i];
	}
	output[i_index] = ans;
}

__global__ void
compute_squared_distances(double *M, double *compute_squared_distances_matrix, double *loghyper, int n, int dim) {
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

/*	if(i_index >= n * n) return;

	int M_row, M_col;
	double ell_sq = exp(loghyper[0] * 2); //l^2 after coverting back from the log form

	M_row = i_index / n;
	M_col = i_index % n;

	if (M_row < M_col) // upper triangular bye bye
		return;

	if (M_row == M_col)
	{
		compute_squared_distances_matrix[M_row * n + M_col] = 0.0;
		return;
	}
	double dot_product = 0.0;
	for (int i = 0; i < dim; i++)
		dot_product += (M[M_row * dim + i] - M[M_col * dim + i]) * (M[M_row * dim + i] - M[M_col * dim + i]);

	compute_squared_distances_matrix[M_row * n + M_col] = compute_squared_distances_matrix[M_col * n + M_row] = dot_product / ell_sq;

*/
	if(i_index >= n || j_index >= n || i_index > j_index) return;
	double ell_sq = exp(loghyper[0] * 2); //l^2 after coverting back from the log form
	if(i_index == j_index){
		compute_squared_distances_matrix[j_index * n + i_index] = 0.0;
		return;
	}	
	double dot_product = 0.0;
	for (int i = 0; i < dim; i++){
		double val1 = M[i_index * dim + i];
		double val2 = M[j_index * dim + i];
		dot_product += (val1 - val2) * (val1 - val2);
	}
	dot_product /= ell_sq;
	compute_squared_distances_matrix[i_index * n + j_index] = dot_product;
	compute_squared_distances_matrix[j_index * n + i_index] = dot_product;
}

__inline__ int upit(int x, int y) {
	return (x + y - 1) / y;
}


void init_and_print()
{
	int deviceCount = 0;
	bool isFastGPU = false;
	std::string name;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	printf("---------------------------------------------------------\n");
	printf("Initializing CUDA for cuGP\n");
	printf("Found %d CUDA devices\n", deviceCount);

	for (int i=0; i<deviceCount; i++) {
		cudaDeviceProp deviceProps;
		cudaGetDeviceProperties(&deviceProps, i);
		name = deviceProps.name;
		if (name.compare("GeForce GTX 480") == 0
				|| name.compare("GeForce GTX 670") == 0
				|| name.compare("GeForce GTX 780") == 0)
		{
			isFastGPU = true;
		}

		printf("Device %d: %s\n", i, deviceProps.name);
		printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
		printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
		printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
	}
	printf("---------------------------------------------------------\n");
	if (!isFastGPU)
	{
		printf("WARNING: "
				"You're not running on a fast GPU, please consider using "
				"NVIDIA GTX 480, 670 or 780.\n");
		printf("---------------------------------------------------------\n");
	}
}

void read_input_and_copy_to_GPU(int numtrain, std::string inputfilename, std::string outputfilename)
{
	printf("Inside read_input_and_copy_to_GPUs\n");
	FILE *input_file, *label_file;
	double *X_host; //input dataset in host!
	double *labels_host; //labels in host!
	lh_host = new double[3];

	input_file = fopen(inputfilename.c_str(), "r");
	label_file = fopen(outputfilename.c_str(), "r");

	fscanf(input_file, "%d%d", &totalN, &DIM);

	for (int i = 0 ; i < 3 ; i++)
		lh_host[i] = 0.5;	

	
	X_host = new double[totalN * DIM];
	labels_host = new double[totalN];

	N = numtrain;

	printf("Reading inputs boy\n");
	printf("Number of inputs = %d\n", totalN);
 
	// Reading inputs
	for (int i = 0; i < totalN; i++)
		for (int j = 0; j < DIM; j++)
			fscanf(input_file, "%lf", &X_host[i * DIM + j]);

	// Reading labels (target values)
	for (int i = 0; i < totalN; i++) {
                fscanf(label_file, "%lf", &labels_host[i]);
        }
	
	printf("reading labels \n");
	cudacall(cudaMalloc(&X, sizeof(double) * totalN * DIM));
	cudacall(cudaMemcpy(X, X_host, sizeof(double) * totalN * DIM, cudaMemcpyHostToDevice));	

	cudacall(cudaMalloc(&labels, sizeof(double) * totalN ));
	cudacall(cudaMemcpy(labels, labels_host, sizeof(double) * totalN , cudaMemcpyHostToDevice));	
	
	cudacall(cudaMalloc(&loghyper, sizeof(double) * 3));
	cudacall(cudaMemcpy(loghyper, lh_host, sizeof(double) * 3 , cudaMemcpyHostToDevice));	
	
	printf("Okay boy.. reading and malloc done\n\n");
}

void setup_loglikelihood_data()
{
	// this is the covariance matrix
	cudacall(cudaMalloc(&K, sizeof(double) * N * N));
	
	//matrix for storing K.inverse()
	cudacall(cudaMalloc(&Kinv, sizeof(double) * N * N));

	//matrix for storing K.inverse()
	cudacall(cudaMalloc(&lowermat_inv_store, sizeof(double) * N * N));
	
	// this is the log determinant
	cudacall(cudaMalloc(&log_det, sizeof(double)));

	// this is for the dot product
	cudacall(cudaMalloc(&ll_dotprod, sizeof(double)));
	
	cudacall(cudaMalloc(&temp_fs, sizeof(double) * N));
	cudacall(cudaMalloc(&temp_bs, sizeof(double) * N));
		
	// just for checking cholesky correctness, delete later FIXME
	orig_sym = new double[N * N]; // should be equal to covariance matrix
}

void setup_cholesky(int, int);

void setup_gradienthp_data(){
	int threads_per_block, number_of_blocks;
	
	// this is the squarred distance matrix
	cudacall(cudaMalloc(&Ksqdist, sizeof(double) * N * N));
	
	cudacall(cudaMalloc(&matforell, sizeof(double) * N * N));

	// CHECK: if we can get away without this
	cudacall(cudaMalloc(&identity, sizeof(double) * N *N));
	cudacall(cudaMemset((void *)identity, 0.0, sizeof(double) * N * N));

	//matrix for storing forward substitution result
	cudacall(cudaMalloc(&tempfsforkinv, sizeof(double) * N *N));
		
	
	// Kintact = K (From compute_K_train)
	cudacall(cudaMalloc(&Kintact, sizeof(double) * N * N));
	
	cudacall(cudaMalloc(&temp1dvec, sizeof(double) * N));
	
	cudacall(cudaMalloc(&tempdiagonal, sizeof(double) * N));

	cudacall(cudaMalloc(&tempWmatrix, sizeof(double) * N * N));

	
}

void setup_TMI()
{
	cudacall(cudaMalloc(&tmi_intermediate_output, sizeof(double) * N * N));
}

void destruct_cublas_cusoler(){
	cusolverDnDestroy(solver_handle);
	cublasDestroy ( blas_handle );
}
void setup_cublas_cusolver(){
		
        cudacall(cudaMalloc(&devInfo, sizeof(int)));
        cusolverDnCreate(&solver_handle);
	stat = cublasCreate (& blas_handle );
}
void setup( int numtrain, std::string inputfilename, std::string outputfilename)
{
	printf("YEEEEEEEEEEEEEEEHH setup call huaa\n");
	std::string s;
	read_input_and_copy_to_GPU(numtrain, inputfilename, outputfilename);

	setup_loglikelihood_data();

	setup_cholesky(N, 2);
	
	setup_gradienthp_data();	

	setup_TMI();
	
	setup_cublas_cusolver();
}

void setup_cholesky(int dim, int b)
{
	/* cudacall(cudaMalloc(&mt, sizeof(double) * dim * dim));
	   cudacall(cudaMalloc(&mt_transpose, sizeof(double) * dim * dim));
	   cudacall(cudaMemcpy(mt, temp_m, sizeof(double) * dim * dim, cudaMemcpyHostToDevice)); */

	/*
	 * Now malloc the a11 matrix
	 */

	cudacall(cudaMalloc(&a11, sizeof(double) * 4));

	/*
	 * Now malloc the a21_transpose matrix by overprovisioning. This can be of maximum size bx(dim-b). But, we allocate
	 * a bx(dim-b) vector even for the latter stages.
	 */

	cudacall(cudaMalloc(&a21_transpose, sizeof(double) * b * (dim - b)));
	cudacall(cudaMemset((void *)a21_transpose, 0, sizeof(double) * b * (dim - b)));

	/*
	 * Now malloc the l21_transpose_from_fs matrix to insert the output of forward substitution. Is retained here for generating a22.
	 */

	cudacall(cudaMalloc(&l21_transpose_from_fs, sizeof(double) * b * (dim - b)));
	cudacall(cudaMemset((void *)l21_transpose_from_fs, 0, sizeof(double) * b * (dim - b)));

	/*
	 * Now malloc the l21 matrix, which will be useful for populating a22 (via matrix mult).
	 */

	cudacall(cudaMalloc(&l21, sizeof(double) * b * (dim - b)));

	/*
	 * Now malloc the l22_temp matrix, which will be useful for elementwise subtraction for a22 (after matrix mult).
	 */

	cudacall(cudaMalloc(&l22_temp, sizeof(double) * (dim - b) * (dim - b)));

}



// M is a device pointer!!, n is the size of the matrix
void get_cholesky_using_cublas(double *M, int n){

	printf("Haaaar REE call hua\n");
	double startime = CycleTimer::currentSeconds();	
        int work_size = 0;


	//print_matrix_kernel<<<1,1>>>(M, n, n);
	//cudaThreadSynchronize(); 
        
	// --- CUDA CHOLESKY initialization: Not needed
         cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, n, M, n * n, &work_size);
        
	// --- CUDA POTRF execution
	 double *work;
  	 cudacall(cudaMalloc(&work, work_size * sizeof(double)));
   	 cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, n, M, n, work, work_size, devInfo);

	// Giving the entire Kinv for buffer: Be happy!!
        //cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, n, M, n, Kinv, n * n, devInfo);
	
       int devInfo_h = 0;
       cudacall(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) std::cout   << "Unsuccessful potrf execution, value = " << devInfo_h << std::endl;
	else printf("okay cholesky went fine\n");

	double endtime = CycleTimer::currentSeconds();
	printf("Time taken for CUSOLVER cholesky: %lf\n", endtime - startime);
	//now make upper vala 0
/*	int threads_per_block = 1024;
        int number_of_blocks = upit( (n * n), threads_per_block);
        set_upper_zero<<<number_of_blocks, threads_per_block>>>(M, n);
        cudaThreadSynchronize();
*/

	dim3 blockDim(32,32);
 	dim3 gridDim( upit(n, blockDim.x), upit(n, blockDim.y));
        set_upper_zero<<<gridDim, blockDim>>>(M, n);
        cudaThreadSynchronize();

}


void get_inverse_by_cublas(double *Lmat, int sizelmat){
        double al =1.0f;

	int threads_per_block, number_of_blocks;
	//threads_per_block = 1024;
	//number_of_blocks = upit((sizelmat * sizelmat), threads_per_block);
	//generate_identity<<<number_of_blocks, threads_per_block>>>(lowermat_inv_store, sizelmat);

	dim3 blockDim(32,32);
 	dim3 gridDim( upit(N, blockDim.x), upit(N, blockDim.y));
	generate_identity<<<gridDim, blockDim>>>(lowermat_inv_store, sizelmat);
	cudaThreadSynchronize(); 
		
        double startime = CycleTimer::currentSeconds();
        (cublasDtrsm(blas_handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT, sizelmat , sizelmat ,&al, Lmat, sizelmat, lowermat_inv_store, sizelmat));
        double endtime = CycleTimer::currentSeconds();
	printf("time taken by cublas-TMI-inverse = %lf\n", endtime - startime);
}

void matrix_multiply_cublas_withtranspose(double *A, double *B, double *C, int size){
     const double alf = 1;
     const double bet = 0;
     const double *alpha = &alf;
     const double *beta = &bet;

     // Do the actual multiplication
     cublasDgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T, size, size, size, alpha, A, size, B, size, beta, C, size);
}

void matrix_vector_multiply_cublas(double *A, double *B, double *C, int size){
     const double alf = 1;
     const double bet = 0;
     const double *alpha = &alf;
     const double *beta = &bet;

     // Do the actual multiplication
     cublasDgemv(blas_handle, CUBLAS_OP_N, size, size, alpha, A, size, B, 1, beta, C, 1);
}



void get_inverse_by_tmi(double *, int );
void compute_chol_get_mul_and_det()
{
	int threads_per_block, number_of_blocks;


//	get_cholesky(K, N); // set of kernels : Now K is actually L
	get_cholesky_using_cublas(K, N); // set of kernels

//	printf("INVERSE KE BAAD\n");
//	print_matrix_kernel<<<1,1>>>(K, N, N);
//	cudaThreadSynchronize(); 

	get_determinant_from_L<<<1, 1>>>(K, N, log_det);
	cudaThreadSynchronize();
	
 	/* // forward_solve_vector(); // kernel K * y = target -> solves for y (Note K is a lower triangular matrix)
	threads_per_block = 512;
	number_of_blocks = upit(N, threads_per_block);
	forward_substitution_vector<<<1, 1>>>(K, labels, temp_fs, N);
	cudaThreadSynchronize();

	// backward_solve_vector(); // kernel L_transpose * x = y -> solves for x 
	backward_substitution_vector<<<1, 1>>>(K, temp_fs, temp_bs, N); // Since we use the K.transpose() inside we don't pass L_transpose
	cudaThreadSynchronize();
	
	// compute_product(); // kernel
	vector_dot_product<<<1, 1>>>(temp_bs, labels, ll_dotprod, N);
	cudaThreadSynchronize();

	//LMI of K
	//vector-matrix-mulitply of (ouput of prev step) and labels => temp_bs
	//TODO: CAN DELETE TEMP_FS..... */

	printf("okay NOW TMI with cublas instead of our call\n");
//	get_inverse_by_tmi(K, N);
//     	cudaThreadSynchronize();
	get_inverse_by_cublas(K, N); //-> Now note that the result will be in lowermat_inv_store and not K (IMP)


//      NOTE: THIS TRANPOSE IS NOT REQUIRED
//		BECAUSE WE HAVE AN ADHOC MATRIX_MULITPLY_CUBLAS_WITHTRANPOSE
//	double alpha = 1.0, beta = 0.0;
//	//now we need to get the transpose lowermat_inv_store : let's store this in K (reuse!!!)
//	cublasDgeam(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, &alpha, lowermat_inv_store, N, &beta, lowermat_inv_store, N, K, N);	

//	printf("Bhai transpose dekhle \n");
//	print_matrix_kernel<<<1, 1>>>(K,N, N);
//	cudaThreadSynchronize(); 

	//Now we will do the DGEMM matrix multiply	
//	printf("SIMPLE MULTIPLY\n");
//	dim3 blockDim(32,32);
 //       dim3 gridDim( upit(N, blockDim.x), upit(N, blockDim.y));
  //      lowertriangular_matrixmultiply_noshare<<<gridDim, blockDim >>>(lowermat_inv_store, Kinv, N);
  //      cudaThreadSynchronize();
	
	matrix_multiply_cublas_withtranspose(lowermat_inv_store, lowermat_inv_store, Kinv, N);
	//printf("Bhai DGEMM dekhle \n");
	//print_matrix_kernel<<<1, 1>>>(Kinv,N, N);
	//cudaThreadSynchronize(); 
	
	

	matrix_vector_multiply_cublas(Kinv, labels, temp_bs, N);
	//printf("\nokay neeche dekho\n");
	//print_vector<<<1, 1>>>(temp_bs, N);
	//cudaThreadSynchronize(); 

	/*	
        threads_per_block = 512;
        number_of_blocks = upit(N, threads_per_block);
        matrix_vector_multiply<<<number_of_blocks, threads_per_block>>>(Kinv, labels, temp_bs, N);
        cudaThreadSynchronize();

	printf("\n\nokay DOT PRODUCT YEH AYA HAI\n");
	print_vector<<<1, 1>>>(temp_bs, N);
	cudaThreadSynchronize(); 
	
	vector_dot_product<<<1, 1>>>(temp_bs, labels, ll_dotprod, N);
	cudaThreadSynchronize();
	*/

	

	thrust::device_ptr<double> td1 = thrust::device_pointer_cast(temp_bs);
	thrust::device_ptr<double> td2 = thrust::device_pointer_cast(labels);

	double ans = 0.0;
	ans = thrust::inner_product(td1, td1 + N, td2, 0.0);
	printf("\n\n ############# PRODUCT YEH AYA = %lf\n", ans);
	//POTENTIAL IMPROVEMENT POSSIBLE, can return ans; instead of doing transfer shittt, 
	cudacall(cudaMemcpy(ll_dotprod, &ans, sizeof(double), cudaMemcpyHostToDevice));

}


double evaluate_and_get_log_likelihood(){
	double term1_ll;
	double term2_ll;
	cudacall(cudaMemcpy(&term1_ll, ll_dotprod,  sizeof(double), cudaMemcpyDeviceToHost));
	cudacall(cudaMemcpy(&term2_ll, log_det ,  sizeof(double), cudaMemcpyDeviceToHost));
	printf("product vala term = %lf\n", term1_ll);
	return -0.5 * ( term1_ll + term2_ll + N * 1.83787);
}
double compute_log_likelihood()
{
	int threads_per_block, number_of_blocks;

	double compute_ll_start = CycleTimer::currentSeconds();

	printf("compute_K_train hota hai\n");

	dim3 blockDim1(32,32);
 	dim3 gridDim1( upit(N, blockDim1.x), upit(N, blockDim1.y));
	double ckts = CycleTimer::currentSeconds();
	compute_K_train<<<gridDim1, blockDim1>>>(X, K, loghyper, N, DIM); // kernel
	cudaThreadSynchronize();
	double ckte = CycleTimer::currentSeconds();
	printf("compute_K_train time: %lf\n", ckte - ckts);

	compute_chol_get_mul_and_det(); // set of kernels

	double llans = evaluate_and_get_log_likelihood(); // kernel, or can be clubbed somewhere
	printf("The value of loglikelihood = %lf\n", llans);
	double compute_ll_end = CycleTimer::currentSeconds();
	printf("compute_log_likelihood Time: %lf\n", compute_ll_end - compute_ll_start);
	return llans; 
}

void compute_K_inverse_using_cublas()
{
	int threads_per_block, number_of_blocks;
	
	// make_identity(); -> did this in setup "identity" is a double *
	double st = CycleTimer::currentSeconds();

	// SUB: get_cholesky(K, N); //Set of kernels, the answer (a lower triangular matrix) is stored 
	double chol_st = CycleTimer::currentSeconds();
	get_cholesky_using_cublas(K, N); // NOW result is in K
	double chol_end = CycleTimer::currentSeconds();
	printf("get_cholesky_using_cublas time: %lf\n", chol_end - chol_st);

	/* SUB:
	threads_per_block = 512;
	number_of_blocks = upit(N, threads_per_block);
	forward_substitution_matrix<<<number_of_blocks, threads_per_block>>>(K, identity, tempfsforkinv, N); // kernel - need N threads
	cudaThreadSynchronize();
	
	// matrix_transpose(); // kernel - Not NEEDED

	// matrix_backward_substitution();
	backward_substitution_matrix<<<number_of_blocks, threads_per_block>>>(K, tempfsforkinv, Kinv, N); // kernel - need N threads
	cudaThreadSynchronize();
	*/

	get_inverse_by_cublas(K, N); //-> Now note that the result will be in lowermat_inv_store and not K (IMP)

	matrix_multiply_cublas_withtranspose(lowermat_inv_store, lowermat_inv_store, Kinv, N);
	double end = CycleTimer::currentSeconds();
	printf("compute_K_inverse_using_cublas time: %lf\n", end - st);
	//-> So the answer is in Kinv
}

void compute_gradient_log_hyperparams(double *localhp_grad)
{
	double start = CycleTimer::currentSeconds();
	int threads_per_block, number_of_blocks;

	double *tt = get_loghyperparam(); //just for a MEMCPY from device to host
	double noise_var = exp(lh_host[2] * 2); //noise variance


	dim3 blockDim1(32,32);
 	dim3 gridDim1( upit(N, blockDim1.x), upit(N, blockDim1.y));
        compute_K_train<<<gridDim1, blockDim1>>>(X, K, loghyper, N, DIM); // kernel
        cudaThreadSynchronize();

	dim3 blockDim2(32,32);
 	dim3 gridDim2( upit(N, blockDim2.x), upit(N, blockDim2.y));
	copy_Kmatrix<<<gridDim2, blockDim2>>>(K, Kintact, N);
        cudaThreadSynchronize();
	
	dim3 blockDim3(32,32);
 	dim3 gridDim3( upit(N, blockDim3.x), upit(N, blockDim3.y));
   	compute_squared_distances<<<gridDim3, blockDim3>>>(X,  Ksqdist,  loghyper,  N, DIM);
   	cudaThreadSynchronize();

	dim3 blockDim4(32,32);
 	dim3 gridDim4( upit(N, blockDim4.x), upit(N, blockDim4.y));
	elementwise_matrix_mult<<<gridDim4, blockDim4>>>(K, Ksqdist, matforell, N, N);
	cudaThreadSynchronize();

	compute_K_inverse_using_cublas(); // set of kernels

	matrix_vector_multiply_cublas(Kinv, labels, temp1dvec, N); //can use temp_bs only

	dim3 blockDim5(32,32);
 	dim3 gridDim5( upit(N, blockDim5.x), upit(N, blockDim5.y));
	copy_Kmatrix<<<gridDim5, blockDim5>>>(Kinv, tempWmatrix, N);
        cudaThreadSynchronize();


	const double alf = -1; //NOTE: we neeed to subtract, that's why -1
     	const double *alpha = &alf;

	cublasDger(blas_handle, N, N, alpha,  temp1dvec, 1, temp1dvec, 1, tempWmatrix, N);

        threads_per_block = 1024;
        number_of_blocks = upit( N, threads_per_block);
	gather_diagonal<<<number_of_blocks, threads_per_block>>>(tempWmatrix, tempdiagonal, N);	
      	cudaThreadSynchronize();
	
	// Now update_log_hyperparams(); 
	thrust::device_ptr<double> td1 = thrust::device_pointer_cast(tempWmatrix);
	thrust::device_ptr<double> td2 = thrust::device_pointer_cast(matforell);
	thrust::device_ptr<double> td3 = thrust::device_pointer_cast(Kintact);

	thrust::device_ptr<double> td4 = thrust::device_pointer_cast(tempdiagonal);

	double para1 = 0.0, para2 = 0.0, para3 = 0.0;

	para1 = thrust::inner_product(td1, td1 + N*N, td2, 0.0);
	para2 = thrust::inner_product(td1, td1 + N*N, td3, 0.0);
	para2 = para2 * 2;
	printf("Why yaar? para2 = %lf\n", para2);
	double common_val = thrust::reduce(td4, td4 + N);
	common_val *= noise_var * 2;

	para3 = common_val;
	para2 = para2 - common_val;

	localhp_grad[0] = para1/2.0;
	localhp_grad[1] = para2/2.0;
	localhp_grad[2] = para3/2.0;
	printf("Dekho bhai %lf\n%lf\n%lf\n", localhp_grad[0], localhp_grad[1], localhp_grad[2]);
	double end = CycleTimer::currentSeconds();
	printf("compute_gradient_hyperparam time: %lf\n", end - start);

}


double *get_loghyperparam(){
	cudacall(cudaMemcpy(lh_host, loghyper,  sizeof(double) * 3, cudaMemcpyDeviceToHost));	
	return  lh_host;
}

void set_loghyper_eigen(Eigen::VectorXd initval) {
        for(int i = 0 ; i < 3; i++) { 
                lh_host[i] = initval[i];
        }
	printf("Dekho bhai naya value AYAA\n\n");
	for(int i = 0 ; i < 3 ; i++){
		printf("%lf\n", lh_host[i]);
	}
	//Now dump it back to loghyper
	cudacall(cudaMemcpy(loghyper, lh_host, sizeof(double) * 3 , cudaMemcpyHostToDevice)); 
}


void setup_for_testing(int offset, int numtest){
	Xtest = X + DIM * offset;
	labelstest = labels + offset;
	Ntest = numtest;	
	
	cudacall(cudaMalloc(&tmeanvec, sizeof(double) * Ntest));
	cudacall(cudaMalloc(&tvarvec, sizeof(double) * Ntest));
	
	//Remember Ktest_vec should have size = N, not Ntest
	cudacall(cudaMalloc(&Ktest_vec, sizeof(double) * N));

}

//compute_test_means_and_variances is a set of kernels: FIX testing code
void compute_test_means_and_variances(){
	int threads_per_block, number_of_blocks;

	//Maybe can move the compute_K_train to setup in SCHEDULER-vala (THINK ABOUT IT SID)
        threads_per_block = 1024;
        number_of_blocks = upit((N * N), threads_per_block);
    	// FIXME
	//    compute_K_train<<<number_of_blocks, threads_per_block>>>(X, K, loghyper, N, DIM); // populated in K
        cudaThreadSynchronize();
	
	//compute_K_inverse(); //populates Kinv with K.inverse()
	// instead of compute_K_inverse, let's see if TMI is of help!!!
	
	//FIXME: cublas vala call KAROOOOOOOOOOOOOOO
	//compute_K_inverse_with_tmi();	
	
	// vector_Kinvy_using_cholesky(); // set of kernels
	// We don't need this: we already have Kinv, so we just need to multiply Kinv and y
        threads_per_block = 512;
        number_of_blocks = upit( N, threads_per_block);
	matrix_vector_multiply<<<number_of_blocks, threads_per_block>>>(Kinv, labels, temp1dvec, N); //so temp1dvec gets populated
      	cudaThreadSynchronize();

	
	double sig_var = exp(lh_host[1] * 2); //signal variance
	double noise_var = exp(lh_host[2] * 2); //noise variance
	for(int i = 0; i < Ntest; i++){

        	threads_per_block = 512;
	        number_of_blocks = upit( N, threads_per_block);
		compute_K_test<<<number_of_blocks, threads_per_block>>>(X, Xtest + i * DIM, Ktest_vec, loghyper, N, DIM); //REUSE SOME ALREADY EXISTING MALLOC'ED 1D VECTOR
      		cudaThreadSynchronize();

		//FIXME
		//vector_dot_product<<<1, 1>>>(Ktest_vec, temp1dvec, tmeanvec + i, N); //for mean
      		//cudaThreadSynchronize();
	
		//FIXME
		//threads_per_block = 512;
	       // number_of_blocks = upit(N, threads_per_block);
       		//vector_matrix_multiply<<<number_of_blocks, threads_per_block>>>(Ktest_vec, Kinv, temp_fs, N); //REUSING temp_fs from likelihood computation
        //	cudaThreadSynchronize();
		//FIXME		
//		vector_dot_product_with_loghp<<<1, 1>>>(Ktest_vec, temp_fs, tvarvec + i, N, sig_var, noise_var ); //for variance
      //		cudaThreadSynchronize();
		
	}
}

void get_negative_log_predprob(){
	
	double *finalans = new double; //for host
	double *ans_nlpp;
	cudacall(cudaMalloc(&ans_nlpp, sizeof(double) ));
	
	compute_NLPP<<<1, 1>>>(labelstest, tmeanvec, tvarvec, Ntest, ans_nlpp);
      	cudaThreadSynchronize();
	
	cudacall(cudaMemcpy(finalans, ans_nlpp,  sizeof(double), cudaMemcpyDeviceToHost));
	printf("OKAY FINAL NLPP = %lf\n", *finalans);
}

void testing_phase(int offset, int numtest){

	printf("---------------------------------------\n");
	printf("TRYING TO START TESTING PHASE\n");	
	printf("---------------------------------------\n");
	//setup THINGS
	setup_for_testing(offset, numtest);
	
	printf("\n---------------------------------------\n");
	printf("TRYING TO Start compute_test_means PHASE\n");	
	printf("---------------------------------------\n");
	// Now calling testing phase
	double startime = CycleTimer::currentSeconds();	
	compute_test_means_and_variances();
	double endtime = CycleTimer::currentSeconds();
	printf("TESTING TIME = %lf\n", endtime - startime);
		
	printf("\n---------------------------------------\n");
	printf("Now result time\n");	
	printf("---------------------------------------\n");
	// actual answer time
	get_negative_log_predprob();
	
}
