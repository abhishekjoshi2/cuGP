#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdlib>
#include "../common/cycleTimer.h"
#include <fstream>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include "./Eigen/Dense"
#include "./Eigen/src/Core/util/DisableStupidWarnings.h"
#include <cmath>
#include<string>

#define INPUT_FILE "../cpp_serial_gp/sine_256_input.txt"
#define LABEL_FILE "../cpp_serial_gp/sine_256_labels.txt"


#define BLOCK_SIZE 32

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
		//printf("predvar = %lf, predmean = %lf, actualmean = %lf, lpp = %lf\n", predicted_testvar[i], predicted_testmean[i], actualtestlabels[i], val);
                ans += val;
        }
	//printf("TO FINAL ANSWER YEH HONA CHAHHIYE: %lf\n", ans / Ntest);
        *ans_nlpp = (ans / Ntest);

}
__global__ void lowertriangular_matrixmultiply_noshare(double *a, double *output, int size)
{

        long long int row = blockIdx.y * blockDim.y + threadIdx.y;
        long long int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= size || col >= size)
                return;

        double sum = 0.0;
        for (int i = 0; i < size; i++)
        {
                //sum += a[row * colsA + i] * b[i * colsB + col]; 
                sum += a[i * size + row] * a[i * size + col];
        }

        output[row * size + col] = sum;
}

//FIXME: can do a shared memory reduce
__global__ void vector_dot_product_with_loghp(double *x1, double *x2, double *ans, int N, double noisevar, double sigvar){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= 1)
		return;

	double val = 0.0;
	for(int i = 0; i < N; i++) {
		val += (x1[i]*x2[i]);
	}
	*ans = -val + noisevar + sigvar;
}

//FIXME: can do a shared memory reduce
__global__ void vector_dot_product(double *x1, double *x2, double *ans, int N){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= 1)
		return;

	double val = 0.0;
	for(int i = 0; i < N; i++) {
		val += (x1[i]*x2[i]);
	}
	*ans = val;
}

__global__ void copy_Kmatrix(double *input, double *output, int size){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= size * size)
		return;
	output[i_index] = input[i_index];
}
__global__ void gather_diagonal(double *inputMat, double *diagvec, int size){
	
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= size)
		return;
	diagvec[i_index] = inputMat[i_index * size + i_index];

}
__global__ void make_identity(double *identity, int N){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= N)
		return;
	identity[i_index * N + i_index] = 1.0;
}

__global__ void outerprod_and_subtract(double *Mat1, double *vec, double *outputMat, int size){
	
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= size * size)
		return;
	
	int i = i_index / size;
	int j = i_index % size;
	
	outputMat[i*size + j] = Mat1[i*size + j] - vec[i]*vec[j];

}
__global__ void check_forward_sub_vector(double *L, double *x, double *b, int N){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= 1)
		return;

	for(int i = 0; i < N; i++) {
		double temp = 0.0;
		for(int j = 0; j < N; j++) {
			temp += L[i*N + j] * x[j];
		}
		printf("%lf - %lf = %lf\n", temp, b[i], temp - b[i]);
	}
}
__global__ void check_backward_sub_vector(double *L, double *y, double *b, int N){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= 1)
		return;

	for(int i = 0; i < N; i++) {
		double temp = 0.0;
		for(int j = 0; j < N; j++) {
			temp += L[j * N + i] * y[j];
		}
		printf("%lf - %lf = %lf\n", temp, b[i], temp - b[i]);
	}
}


// We want to solve for "output", such that 
//	 lowert_mat * output = b;
__global__ void forward_substitution_vector(double *lowert_mat, double *b, double *output, int N){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= 1)
		return;

	/*
	// Forward solve L * temp = y
	for (int i = 0; i < n; i++){
	temp[i] = y[i];
	for (int j = 0; j < i; j++)
	{
	temp[i] -= L[i][j] * temp[j];
	}
	temp[i] /= L[i][i];
	}
	 */
	for(int i = 0 ; i < N ; i++){
		output[i] = b[i];	
		for (int j = 0 ; j < i; j++){
			output[i] -= lowert_mat[i * N + j] * output[j];
		}
		output[i] /= lowert_mat[i*N + i];
	}
}


// We want lower_mat.transpose() * output = b;
// NOTE: WE ARE NOT PASSING AN UPPER TRIANGULAR MATRIX (which would have been the case in a general implementation)
__global__ void backward_substitution_vector(double *lowert_mat, double *b, double *output, int N){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= 1)
		return;

	/*
	// backward solve arr2 * y2 = b
	for(int i = DIM - 1; i >= 0 ; i--) {
	y2[i] = b[i];
	for(int j = i + 1; j < DIM; j++) {
	y2[i] = y2[i] - arr2[i][j] * y2[j];
	}
	y2[i] = y2[i] / arr2[i][i];
	}
	 */

	for(int i = N - 1; i >= 0 ; i--) {
		output[i] = b[i];
		for(int j = i + 1; j < N; j++) {
			output[i] = output[i] - lowert_mat[j * N + i] * output[j];
		}
		output[i] /= lowert_mat[i*N + i];
	}
}

// We want: A.transpose() * output = B
// NOTE: WE ARE NOT PASSING AN UPPER TRIANGULAR MATRIX (which would have been the case in a general implementation)
__global__ void backward_substitution_matrix(double *A, double *B, double *output, int size){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	if (i_index >= size)
		return;

	 /*
	 for (int k = 0; k < DIM; k++) {
                for (int i = DIM - 1; i >= 0 ; i--) {
                        output[i][k] = B[i][k];
                        for (int j = i + 1; j < DIM; j++) {
                                output[i][k] = output[i][k] - A[i][j] * output[j][k];
                        }
                        output[i][k] = output[i][k] / A[i][i];
                }
        }
	*/
	int k = i_index;

	for(int i = size - 1; i >= 0 ; i--) {
		output[i*size + k] = B[i*size + k];
		for(int j = i + 1; j < size; j++) {
			// This is when A was proper: output[i*size + k] -= A[i*size + j] * output[j*size + k];  
			// But we have to take A.transpose(), so we swap i,j 
			output[i*size + k] -= A[j*size + i] * output[j*size + k];  
		}
		output[i*size + k] /= A[i*size + i]; //No need to change here, as only diagonal elements are accessed
	}
}

__global__ void set_upper_zero(double *M, int dim){

	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= (dim * dim))
		return;

	int rowN = i_index / dim;
	int colN = i_index % dim;

	if(rowN >= colN) return;

	M[rowN * dim + colN] = 0.0;
}

	__global__ void
hardcoded_cholesky_1x1(double *M, double *a11, int dim, int b, int start_id)
{
	// TODO
	/* M[idx][idx] = sqrt(M[idx][idx]);
	   a11[0][0] = M[idx][idx]; */
}

	__global__ void
hardcoded_cholesky_2x2(double *M, double *a11, int dim, int b, int start_id)
{
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	/*
	   printf("In kernel\n");
	   printf("dim is %d, i_index is %d, j_index is %d, b is %d, start_id is %d\n", dim, i_index, j_index, b, start_id);

	   printf("Now the 2x2 matrix is:\n");
	   for (int i = 0; i < b; i++)
	   {
	   for (int j = 0; j < b; j++)
	   {
	   printf("%lf ", M[(i + start_id) * dim + j + start_id]);
	   }
	   printf("\n");
	   }
	 */

	a11[0] = M[start_id * dim + start_id] = sqrt(M[start_id * dim + start_id]);
	a11[1] = M[start_id * dim + start_id + 1] = 0.0;
	a11[2] = M[(start_id + 1) * dim + start_id] = M[(start_id + 1) * dim + start_id] / M[start_id * dim + start_id];
	a11[3] = M[(start_id + 1) * dim + start_id + 1] = sqrt(M[(start_id + 1) * dim + start_id + 1] - M[(start_id + 1) * dim + start_id] * M[(start_id + 1) * dim + start_id]);

	//printf("printing a11 matrix\n");
	//printf("%lf %lf %lf %lf\n", a11[0], a11[1], a11[2], a11[3]);
}

	__global__ void
print_matrix_kernel(double *arr, int dim1, int dim2)
{
	printf("Printing matrix:\n");
	for (int i = 0; i < dim1; i++)
	{
		for (int j = 0; j < dim2; j++)
		{
			printf("%lf\t", arr[i * dim2 + j]);
		}
		printf("\n");
	}
}

__global__ void
take_a21_transpose(double *M, double *a21_transpose, int dim, int b, int start_id) {
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= (dim - b - start_id) * b)
		return;

	//printf("In a21_transpose, i_index is %d, j_index is %d\n", i_index, j_index);

	int input_row, input_col, target_row, target_col, row_offset_by_thread, col_offset_by_thread;

	// TODO replace i_index by a generic construct because it may involve blocks and grids
	row_offset_by_thread = i_index / b;
	input_row = start_id + b + row_offset_by_thread;

	col_offset_by_thread = i_index % b;
	input_col = start_id + col_offset_by_thread;

	target_row = i_index % b;
	target_col = i_index / b;

	a21_transpose[target_row * (dim - b - start_id) + target_col] = M[input_row * dim + input_col];
}

	__global__ void
forward_substitution_rectangular_a21(double *M, double *a11, double *a21_transpose, double *l21_transpose_from_fs, int dim, int b, int start_id)
{
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= (dim - b - start_id))
		return;

	/* for (int k = 0; k < dim2; k++) { // this is looping over columns of B matrix
	   for (int i = 0; i < dim1; i++) {
	   output[i][k] = B[i][k];
	   for (int j = 0; j < i; j++) {
	   output[i][k] = output[i][k] - A[i][j] * output[j][k];
	   }
	   output[i][k] = output[i][k] / A[i][i];
	   }
	   } */
	int k = i_index;
	// TODO experiment with #pragma unroll
	for (int i = 0; i < b; i++)
	{
		l21_transpose_from_fs[i * (dim - b - start_id) + k] = a21_transpose[i * (dim - b - start_id) + k];
		for (int j = 0; j < i; j++)
		{
			l21_transpose_from_fs[i * (dim - b - start_id) + k] -= a11[i * b + j] * l21_transpose_from_fs[j * (dim - b - start_id) + k];
		}
		l21_transpose_from_fs[i * (dim - b - start_id) + k] /= a11[i * b + i];

		//Updating M too!	
		M[ (start_id + b + k) * dim + start_id + i ] = l21_transpose_from_fs[i * (dim - b - start_id) + k];
	}
}

// For A * C = B, remember we are solving for C, so the third argument 
// 	should be the output argument
__global__ void
forward_substitution_matrix(double *A, double *B, double *output, int size)
{
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= size)
		return;

	/* for (int k = 0; k < dim2; k++) { // this is looping over columns of B matrix
	   	for (int i = 0; i < dim1; i++) {
	   		output[i][k] = B[i][k];
	   		for (int j = 0; j < i; j++) {
	   			output[i][k] = output[i][k] - A[i][j] * output[j][k];
	   		}
	   		output[i][k] = output[i][k] / A[i][i];
	   	}
	   } */

	int k = i_index;

	// TODO experiment with #pragma unroll
	for (int i = 0; i < size; i++)
	{
		output[i * size + k] = B[i * size + k];
		for (int j = 0; j < i; j++)
		{
			output[i * size + k] -= A[i * size + j] * output[j * size + k];
		}
		output[i * size + k] /= A[i * size + i];
	}
}


__global__ void
check_l21_kernel(double *M1, double *M2, double* targetoutput, int d1, int d2, int d3){

	double totaldiff = 0.0, diff = 0;
	for(int i = 0; i < d1; i++){
		for(int j = 0; j < d3 ;j++){ 
			double tempval = 0.0;
			for(int k = 0; k < d2; k++){
				//tempval += M1[i][k] + M2[k][j];
				tempval += M1[i*d2 + k] * M2[k * d3 + j];
			}
			//diff = tempval - targetoutput[i][j];
			diff = tempval - targetoutput[i * d3 + j];

			totaldiff += diff * diff;
			//printf("Diff = %lf\n", diff);
		}
	}
	printf("The error for l21_transpose_from_fs is %lf\n", totaldiff);
}
__global__ void
singlethread_temp_matmult_kernel(double *M1, double *M2, double* targetoutput, int d1, int d2, int d3){	
	for(int i = 0; i < d1; i++){
		for(int j = 0; j < d3 ;j++){ 
			double tempval = 0.0;
			for(int k = 0; k < d2; k++){
				tempval += M1[i*d2 + k] * M2[k * d3 + j];
			}
			targetoutput[i * d3 + j] = tempval;
		}
	}
}

__global__ void
generic_matrix_transpose(double *input, double *output, int d1, int d2){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= d1 * d2)
		return;

	// DESIRED: output[j][i] = input[i][j];
	//	    output: d2 x d1
	// 	    input : d1 x d2

	// int input_row = index / d2;
	// int input_col = index % d2;
	output[ (i_index % d2) * d1 + (i_index / d2)]  = input[i_index];
}

__global__ void matrixmultiply_noshare(double *a, int rowsA, int colsA, double *b, int rowsB, int colsB, double *c)
{

	long long int row = blockIdx.y * blockDim.y + threadIdx.y;
	long long int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= rowsA || col >= colsB)
		return;

	//printf("row: %d, col: %d\n", row, col);
	double sum = 0.0;
	for (int i = 0; i < colsA; i++)
	{
		sum += a[row * colsA + i] * b[i * colsB + col]; 
	}

	c[row * colsB + col] = sum;
}

__global__ void offseted_elementwise_subtraction(double *input, int size, double *M, int dim, int b, int start_id){
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= size * size)
		return;

	// int input_row = i_index / size;
	// int input_col = i_index % size;
	// we want M[ input_row + start_id + b, input_col + start_id + b ] -= input[i_index];

	int input_row = i_index / size;
	int input_col = i_index % size;
	M[ (input_row + start_id + b) * dim + (input_col + start_id + b) ] -= input[i_index];

}

	__global__ void
get_determinant_from_L(double *M, int dim, double *log_det)
{
	// single thread

	double ans = 0.0;
	for (int i = 0; i < dim; i++)
		ans += log(M[i * dim + i]);
	ans *= 2;
	*log_det = ans;
	printf("Determinant is %lf\n", ans);
}

	__global__ void
elementwise_matrix_mult(double *mat1, double *mat2, double *mat3, int rows, int cols)
{
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	int target_row, target_col;
	double dot_product = 0.0;

	target_row = i_index / cols;
	target_col = i_index % cols;

	if (target_row >= rows || target_col >= cols)
		return;

	mat3[target_row * cols + target_col] = mat1[target_row * cols + target_col] * mat2[target_row * cols + target_col];
}

__global__ void
compute_K_train(double *M, double *K_output, double *loghyper, int n, int dim) {
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	if (i_index >= n * n) return;

	double ell_sq = exp(loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(loghyper[1] * 2); // signal variance
	double noise_var = exp(loghyper[2] * 2); //noise variance

	int M_row, M_col;
	double dot_product = 0.0;

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

	if(i_index >= n * n) return;

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

}

__global__ void kernelSharedMemMatMult(double *A, int rowsA, int colsA, 
		double *B, int rowsB,  int colsB, 
		double *C)
{
	double tmp = 0.0;
	__shared__ double M_shared[BLOCK_SIZE][BLOCK_SIZE] ;
	__shared__ double N_shared[BLOCK_SIZE][BLOCK_SIZE] ;

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	for(int m = 0; m < (BLOCK_SIZE + colsA - 1)/BLOCK_SIZE; m++) 
	{
		if(m * BLOCK_SIZE + threadIdx.x < colsA && row < rowsA) 
		{
			M_shared[threadIdx.y][threadIdx.x] =  
				A[row * colsA + m * BLOCK_SIZE + threadIdx.x];
		}
		else 
		{
			M_shared[threadIdx.y][threadIdx.x] = 0.0;
		}

		if(m * BLOCK_SIZE + threadIdx.y < rowsB && col < colsB) 
		{
			N_shared[threadIdx.y][threadIdx.x] =  
				B[(m * BLOCK_SIZE + threadIdx.y) * colsB + col];
		} 
		else 
		{
			N_shared[threadIdx.y][threadIdx.x] = 0.0;
		}
		__syncthreads();


		for(int tileIndex = 0; tileIndex < BLOCK_SIZE; tileIndex++) 
		{
			tmp += M_shared[threadIdx.y][tileIndex] * N_shared[tileIndex][threadIdx.x];
		}
		__syncthreads();
	}

	if(row < rowsA && col < colsB) 
	{
		C[((blockIdx.y * blockDim.y + threadIdx.y) * colsB) + 
			(blockIdx.x * blockDim.x) + threadIdx.x] = tmp;
	}
}


__global__ void inplace_lower_inverse_2x2(double *input_mat, int mat_dim) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int start_row, start_col;
	double m11, m22;

	if (row >= mat_dim / 2)
		return;

	// printf("row is %d and col is %d\n", row, col);
	start_row = row * 2;
	start_col = row * 2;

	m11 = input_mat[start_row * mat_dim + start_col];
	m22 = input_mat[(start_row + 1) * mat_dim + start_col + 1];
	input_mat[(start_row + 1) * mat_dim + start_col] = -input_mat[(start_row + 1) * mat_dim + start_col] / (m11 * m22);

	input_mat[start_row * mat_dim + start_col] = 1 / m11;
	input_mat[(start_row + 1) * mat_dim + start_col + 1] = 1 / m22;
}

__global__ void first_offseted_mat_mult(double *orig, int mat_size, double *tmi_playground, int ltm_dim, int total_threads) {
	
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= total_threads)
		return;
	
	int id, mat_num, ele, row_offset, col_offset, internal_row, internal_col, cur_row, cur_col;

	id = row;
	mat_num = id / (mat_size * mat_size);
	ele = id % (mat_size * mat_size);
	row_offset = mat_size * (2 * mat_num + 1);
	col_offset = row_offset - mat_size;
	internal_row = ele / mat_size;
	internal_col = ele % mat_size;
	cur_row = internal_row + row_offset;
	cur_col = internal_col + col_offset;

	double ans  = 0.0;
	for(int i = 0;i < mat_size; i++){
		ans += orig[(row_offset + i)*ltm_dim + cur_col] * orig[cur_row * ltm_dim + col_offset + mat_size + i];
	}
	tmi_playground[cur_row * ltm_dim + cur_col] = ans;
}

__global__ void second_offseted_mat_mult(double *orig, int mat_size, double *tmi_playground, int ltm_dim, int total_threads) {
	
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= total_threads)
		return;
	
	int id, mat_num, ele, row_offset, col_offset, internal_row, internal_col, cur_row, cur_col;

	id = row;
	mat_num = id / (mat_size * mat_size);
	ele = id % (mat_size * mat_size);
	row_offset = mat_size * (2 * mat_num + 1);
	col_offset = row_offset - mat_size;
	internal_row = ele / mat_size;
	internal_col = ele % mat_size;
	cur_row = internal_row + row_offset;
	cur_col = internal_col + col_offset;

	double ans  = 0.0;
	for(int i = 0;i < mat_size; i++){
		ans += tmi_playground[cur_row * ltm_dim + col_offset  + i] * orig[(row_offset - mat_size + i)*ltm_dim + cur_col];
	}
	orig[cur_row * ltm_dim + cur_col] = -1.0 * ans;
}
__inline__ int upit(int x, int y) {
	return (x + y - 1) / y;
}


void get_symmetric_matrix_1d(double *M, double **matrix1, double **matrix2, int dim) {

	srand(time(NULL));
	int setter = 1;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++){
			matrix1[i][j] = rand() % 100 + 1;
			matrix2[j][i] = matrix1[i][j];
		}
	}
	for (int i = 0; i < dim; i++){
		for(int j = 0; j < dim; j++){
			M[i * dim + j ] = 0.0;
			for(int k = 0; k < dim; k++){
				M[i * dim + j] += matrix1[i][k]*matrix2[k][j];
			}
		}
	}
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

	threads_per_block = 512;
        number_of_blocks = upit(N , threads_per_block);
	make_identity<<<number_of_blocks, threads_per_block>>>(identity, N);  	

	//matrix for storing forward substitution result
	cudacall(cudaMalloc(&tempfsforkinv, sizeof(double) * N *N));
		
	//matrix for storing K.inverse()
	cudacall(cudaMalloc(&Kinv, sizeof(double) * N * N));
	
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

void setup( int numtrain, std::string inputfilename, std::string outputfilename)
{
	std::string s;
	read_input_and_copy_to_GPU(numtrain, inputfilename, outputfilename);

	setup_loglikelihood_data();

	setup_cholesky(N, 2);
	
	setup_gradienthp_data();	

	setup_TMI();
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



	/*GlobalConstants params;
	  params.sceneName = sceneName;
	  params.numCircles = numCircles;
	  params.imageWidth = image->width;
	  params.imageHeight = image->height;
	  params.position = cudaDevicePosition;
	  params.velocity = cudaDeviceVelocity;
	  params.color = cudaDeviceColor;
	  params.radius = cudaDeviceRadius;
	  params.imageData = cudaDeviceImageData;

	  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

	// also need to copy over the noise lookup tables, so we can
	// implement noise on the GPU
	int* permX;
	int* permY;
	float* value1D;
	getNoiseTables(&permX, &permY, &value1D);
	cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
	cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
	cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

	// last, copy over the color table that's used by the shading // function for circles in the snowflake demo

	float lookupTable[COLOR_MAP_SIZE][3] = {
	{1.f, 1.f, 1.f},
	{1.f, 1.f, 1.f},
	{.8f, .9f, 1.f},
	{.8f, .9f, 1.f},
	{.8f, 0.8f, 1.f},
	};

	cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

	if(cudaMalloc(&table, sizeof(int) * numCircles * image->width * image->height / 16 / 16) != cudaSuccess )
	printf("The giant malloc failed.\n");

	if (cudaMalloc(&counters, sizeof(int) * image->height * image->width / 16 / 16) != cudaSuccess)
	printf("The counter malloc fialed.\n");

	if (cudaMalloc(&flagarray, sizeof(int) * numCircles) != cudaSuccess)
	printf("The flag array malloc failed.\n");

	cudaMemset((void *)flagarray, 0, sizeof(int) * numCircles); */
}


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

void get_input(int dim){

	double **m1, **m2;

	temp_m = new double[dim * dim];

	m1 = new double *[dim];
	m2 = new double *[dim];

	for (int i = 0; i < dim; i++)
	{
		m1[i] = new double[dim];
		m2[i] = new double[dim];
	}

	get_symmetric_matrix_1d(temp_m, m1, m2, dim);

	printf("Abhi input hua\n");
	/*
	   printf("Generated matrix in host is \n");
	   for (int i = 0; i < dim; i++)
	   {
	   for (int j = 0; j < dim; j++)
	   {
	   printf("%lf ", temp_m[i * dim + j]);
	   }
	   printf("\n");
	   }
	 */

	for(int i = 0 ; i < dim ; i++){
		delete m1[i];
		delete m2[i];
	}
	delete m1;
	delete m2;

}
void initialize_random(int dim){
	temp_m = new double[dim * dim];
	srand(time(NULL));
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++){
			temp_m[i*dim + j] = ((double) rand() / (RAND_MAX));
			//temp_m[i*dim + j] = rand() % 10;
		}
	}

}



void generate_random_vector(double *b, int dim){
	for(int i = 0 ; i < dim ; i++){
		b[i] = rand() % 10;	
	}
}

void get_cholesky(double *, int);
void setup_for_timing_cholesky(int dimp){
	double **m1, **m2;
	
	setup_cholesky(dimp, 2);

	temp_m = new double[dimp * dimp];

	m1 = new double *[dimp];
	m2 = new double *[dimp];

	for (int i = 0; i < dimp; i++)
	{
		m1[i] = new double[dimp];
		m2[i] = new double[dimp];
	}

	get_symmetric_matrix_1d(temp_m, m1, m2, dimp);

	orig_sym = new double[dimp * dimp]; // should be equal to covariance matrix
	double *devM; //device ka banda
	cudacall(cudaMalloc(&devM, sizeof(double) * dimp * dimp));
	cudacall(cudaMemcpy(devM, temp_m,  sizeof(double) * dimp * dimp, cudaMemcpyHostToDevice));
	
	get_cholesky(devM, dimp);
	for(int i = 0 ; i < dimp ; i++){
		delete[] m1[i];
		delete[] m2[i];
	}
	delete[] m1;
	delete[] m2;



}


void get_cholesky(double *M, int n)
{
	int start_id, b;
	int threads_per_block;
	int number_of_blocks;
	int num_iters;
	double startime, endtime;
	int dim = n;

	cudacall(cudaMemcpy(orig_sym, M,  sizeof(double) * dim * dim, cudaMemcpyDeviceToHost));

	start_id = 0;
	b = 2;

	startime = CycleTimer::currentSeconds();	

	num_iters = n / b;

	for (int i = 0; i < num_iters; i++)
	{
		hardcoded_cholesky_2x2<<<1, 1>>>(M, a11, dim, b, start_id);
		cudaThreadSynchronize();

		if (i == num_iters - 1)
			break;

		// TODO optimize a21_transpose, by bypassing it perhaps? Can avoid transpose and manipulate indices inside next kernel
		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id) * b, threads_per_block);
		take_a21_transpose<<<number_of_blocks, threads_per_block>>>(M, a21_transpose, dim, b, start_id);
		cudaThreadSynchronize();

		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id), threads_per_block);
		forward_substitution_rectangular_a21<<<number_of_blocks, threads_per_block>>>(M, a11, a21_transpose, l21_transpose_from_fs, dim, b, start_id);
		cudaThreadSynchronize();

		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id) * b, threads_per_block);
		generic_matrix_transpose<<<number_of_blocks, threads_per_block>>>(l21_transpose_from_fs, l21, b, dim - b - start_id);
		cudaThreadSynchronize();

		//matrixmultiply_noshare<<<(double *a, int rowsA, int colsA, double *b, int rowsB, int colsB, double *c)
		int rowA = (dim - b - start_id) , colA = b, rowB = b , colB = (dim - b - start_id) ;
		dim3 blockDim(32,32);
		dim3 gridDim( upit(colB, blockDim.x), upit(rowA, blockDim.y));
		matrixmultiply_noshare<<<gridDim, blockDim >>>(l21, (dim - b - start_id), b,  l21_transpose_from_fs, b, dim - b - start_id, l22_temp);
		cudaThreadSynchronize();

		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id) * (dim - b - start_id), threads_per_block);
		offseted_elementwise_subtraction<<<number_of_blocks, threads_per_block >>>(l22_temp, dim - b - start_id, M, dim, b, start_id);
		cudaThreadSynchronize();

		start_id += b;
	}
	endtime = CycleTimer::currentSeconds();	
	// Fire a kernel for making upper-triangular as 0.0
	threads_per_block = 512;
	number_of_blocks = upit( (dim * dim), threads_per_block);
	set_upper_zero<<<number_of_blocks, threads_per_block>>>(M, dim);
	cudaThreadSynchronize();

	printf("Total time taken in cholesky = %lf s\n", endtime - startime);	
	// Now checking!

	double *finalans = new double[dim * dim];

	cudacall(cudaMemcpy(finalans, M,  sizeof(double) * dim * dim, cudaMemcpyDeviceToHost));

	check_cholesky(finalans, orig_sym, dim); 
}

void compute_chol_get_mul_and_det_old()
{
	int threads_per_block, number_of_blocks;

	get_cholesky(K, N); // set of kernels

	get_determinant_from_L<<<1, 1>>>(K, N, log_det);
	cudaThreadSynchronize();

	//threads_per_block = 512;
	//number_of_blocks = upit(N * N, threads_per_block);
	//generic_matrix_transpose<<<number_of_blocks, threads_per_block>>>(K, L_transpose, N, N); // FIXME lesser threads possible!
	//cudaThreadSynchronize();

 	// forward_solve_vector(); // kernel K * y = target -> solves for y (Note K is a lower triangular matrix)
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
}

void get_inverse_by_tmi(double *, int );
void compute_chol_get_mul_and_det()
{
	int threads_per_block, number_of_blocks;

	get_cholesky(K, N); // set of kernels : Now K is actually L

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

	get_inverse_by_tmi(K, N);
      	cudaThreadSynchronize();
  	
	dim3 blockDim(32,32);
        dim3 gridDim( upit(N, blockDim.x), upit(N, blockDim.y));
        lowertriangular_matrixmultiply_noshare<<<gridDim, blockDim >>>(K, Kinv, N);
        cudaThreadSynchronize();

        threads_per_block = 512;
        number_of_blocks = upit(N, threads_per_block);
        matrix_vector_multiply<<<number_of_blocks, threads_per_block>>>(Kinv, labels, temp_bs, N);
        cudaThreadSynchronize();

	vector_dot_product<<<1, 1>>>(temp_bs, labels, ll_dotprod, N);
	cudaThreadSynchronize();

	/* thrust::device_ptr<double> td1 = thrust::device_pointer_cast(temp_bs);
	thrust::device_ptr<double> td2 = thrust::device_pointer_cast(labels);

	double ans = 0.0;
	ans = thrust::inner_product(td1, td1 + N, td2, 0.0);

	//POTENTIAL IMPROVEMENT POSSIBLE
	cudacall(cudaMemcpy(ll_dotprod, &ans, sizeof(double), cudaMemcpyHostToDevice)); */
}


double evaluate_and_get_log_likelihood(){
	double term1_ll;
	double term2_ll;
	cudacall(cudaMemcpy(&term1_ll, ll_dotprod,  sizeof(double), cudaMemcpyDeviceToHost));
	cudacall(cudaMemcpy(&term2_ll, log_det ,  sizeof(double), cudaMemcpyDeviceToHost));
	return -0.5 * ( term1_ll + term2_ll + N * 1.83787);
}
double compute_log_likelihood()
{
	int threads_per_block, number_of_blocks;

	threads_per_block = 512;
	number_of_blocks = upit((N * N), threads_per_block);

	compute_K_train<<<number_of_blocks, threads_per_block>>>(X, K, loghyper, N, DIM); // kernel
	cudaThreadSynchronize();

	/* print_matrix_kernel<<<1,1>>>(K, N, N);
	   cudaThreadSynchronize(); */

	compute_chol_get_mul_and_det(); // set of kernels

	double llans = evaluate_and_get_log_likelihood(); // kernel, or can be clubbed somewhere
	printf("The value of loglikelihood = %lf\n", llans);
	return llans; 
}

void compute_K_inverse()
{
	int threads_per_block, number_of_blocks;
	
	// make_identity(); -> did this in setup "identity" is a double *

	get_cholesky(K, N); //Set of kernels, the answer (a lower triangular matrix) is stored 

	threads_per_block = 512;
	number_of_blocks = upit(N, threads_per_block);
	forward_substitution_matrix<<<number_of_blocks, threads_per_block>>>(K, identity, tempfsforkinv, N); // kernel - need N threads
	cudaThreadSynchronize();
	
	// matrix_transpose(); // kernel - Not NEEDED

	// matrix_backward_substitution();
	backward_substitution_matrix<<<number_of_blocks, threads_per_block>>>(K, tempfsforkinv, Kinv, N); // kernel - need N threads
	cudaThreadSynchronize();
	
}

void compute_K_inverse_with_tmi()
{
	int threads_per_block, number_of_blocks;
	
	// make_identity(); -> did this in setup "identity" is a double *

	get_cholesky(K, N); //Set of kernels, the answer (a lower triangular matrix) is stored 

	get_inverse_by_tmi(K, N);

	/*
	threads_per_block = 512;
	number_of_blocks = upit(N, threads_per_block);
	forward_substitution_matrix<<<number_of_blocks, threads_per_block>>>(K, identity, tempfsforkinv, N); // kernel - need N threads
	cudaThreadSynchronize();
	
	// matrix_transpose(); // kernel - Not NEEDED

	// matrix_backward_substitution();
	backward_substitution_matrix<<<number_of_blocks, threads_per_block>>>(K, tempfsforkinv, Kinv, N); // kernel - need N threads
	cudaThreadSynchronize();
	*/
}

/* We don't need this!
void vector_Kinvy_using_cholesky()
{
	// get_cholesky(); // set of kernels

	// matrix_transpose();

	// forward_solve_vector();

	// backward_solve_vector();
}*/

void compute_gradient_log_hyperparams(double *localhp_grad)
{
	int threads_per_block, number_of_blocks;

	double *tt = get_loghyperparam(); //just for a MEMCPY from device to host
	double noise_var = exp(lh_host[2] * 2); //noise variance

	// compute_K_train(); // kernel - can reuse earlier matrix?
        threads_per_block = 512;
        number_of_blocks = upit((N * N), threads_per_block);
        compute_K_train<<<number_of_blocks, threads_per_block>>>(X, K, loghyper, N, DIM); // kernel
        cudaThreadSynchronize();
	
        threads_per_block = 512;
        number_of_blocks = upit((N * N), threads_per_block);
	copy_Kmatrix<<<number_of_blocks, threads_per_block>>>(K, Kintact, N);
        cudaThreadSynchronize();
		
	
	//compute_squared_distance(); // kernel
        threads_per_block = 512;
        number_of_blocks = upit((N * N), threads_per_block);
   	compute_squared_distances<<<number_of_blocks, threads_per_block>>>(X,  Ksqdist,  loghyper,  N, DIM);
   	cudaThreadSynchronize();

	// elementwise_matrix_mult(); // kernel
        threads_per_block = 512;
        number_of_blocks = upit((N * N), threads_per_block);
	elementwise_matrix_mult<<<number_of_blocks, threads_per_block>>>(K, Ksqdist, matforell, N, N);
	cudaThreadSynchronize();
	
	//print_matrix_kernel<<<1,1>>>(matforell, N, N);
      	//cudaThreadSynchronize();

	printf("Yahi hai Kinv\n");
	compute_K_inverse(); // set of kernels
	//print_matrix_kernel<<<1,1>>>(Kinv, N, N);
      	//cudaThreadSynchronize();

	// vector_Kinvy_using_cholesky(); // set of kernels
	// We don't need this: we already have Kinv, so we just need to multiply Kinv and y
        threads_per_block = 512;
        number_of_blocks = upit( N, threads_per_block);
	matrix_vector_multiply<<<number_of_blocks, threads_per_block>>>(Kinv, labels, temp1dvec, N);
      	cudaThreadSynchronize();

	// get_outer_product(); // kernel
	// subtract_matrices(); // kernel
	// -- Combining the above 2 in a single kernel call
        threads_per_block = 512;
        number_of_blocks = upit( N * N, threads_per_block);
	outerprod_and_subtract<<<number_of_blocks, threads_per_block>>>(Kinv, temp1dvec, tempWmatrix, N);
      	cudaThreadSynchronize();

        threads_per_block = 512;
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

}


/* void testing_kernels() {
   printf("Okay called at least\n");
   int threads_per_block;
   int number_of_blocks;

   double *inputdata;
   double *loghyper;
   double *K_output; //for storing the n x n matrix

   printf("n = %d, dim = %d\n", N, DIM);	

   compute_K_train<<<number_of_blocks, threads_per_block >>>(inputdata, K_output, loghyper, N, DIM);

   cudaThreadSynchronize();

   print_matrix_kernel<<<1,1>>>(K_output, N, N);
   cudaThreadSynchronize();

   printf("\nNow printing the squared distance matrix\n");	

   double c = 0.5;

   threads_per_block = 512;
   number_of_blocks = upit( (N * N), threads_per_block);
   compute_squared_distances<<<number_of_blocks, threads_per_block>>>(inputdata,  K_output,  c,  N, DIM);
   cudaThreadSynchronize();
   print_matrix_kernel<<<1,1>>>(K_output, N, N);
   cudaThreadSynchronize();
   return ;
   printf("Abey yahan toh aya\n");	
   int N = 10;	 //Total number of training samples
// run_kernel_cholesky(N);
printf("Call to cholesky khatam hua\n");	


//NOTE: M will now have a lower triangular matrix
print_matrix_kernel<<<1,1>>>(M, N, N);
cudaThreadSynchronize();

double *mat1, *mat2, *mat3;
double mat1_host[16], mat2_host[16], mat3_host[16];

N = 16; // 4x4 matrices
cudacall(cudaMalloc(&mat1, sizeof(double) * N));
cudacall(cudaMalloc(&mat2, sizeof(double) * N));
cudacall(cudaMalloc(&mat3, sizeof(double) * N));

for (int i = 0; i < 16; i++)
mat1_host[i] = i;

for (int j = 16; j < 32; j++)
mat2_host[j - 16] = j;

cudacall(cudaMemcpy(mat1, mat1_host, sizeof(double) * N , cudaMemcpyHostToDevice));	
cudacall(cudaMemcpy(mat2, mat2_host, sizeof(double) * N , cudaMemcpyHostToDevice));	

printf("mat1:\n");
print_matrix_kernel<<<1, 1>>>(mat1, 4, 4);
cudaThreadSynchronize();

printf("mat2:\n");
print_matrix_kernel<<<1, 1>>>(mat2, 4, 4);
cudaThreadSynchronize();

elementwise_matrix_mult<<<1, 256>>>(mat1, mat2, mat3, 4, 4);
cudaThreadSynchronize();

printf("mat3:\n");
print_matrix_kernel<<<1, 1>>>(mat3, 4, 4);
cudaThreadSynchronize();

return ;

double *log_det;
cudacall(cudaMalloc(&log_det, sizeof(double) * 1));
get_determinant_from_L<<<1, 1>>>(M, N, log_det);
cudaThreadSynchronize();
return;	

// FORWARD SUBSTITUTION 
//	
//	generating random targets!!	
double *b = new double[N];

double *labels_vec; //This is the array with the target values in the dataset, it is vector for CUDA
// Next, we will have to load the values from the file instead of copying from b	

double *fsvec; // This is the array that will contain the result of ForwardSubstitutionVector call

generate_random_vector(b, N);	
//	Allocating appropriate memory chunks.
cudacall(cudaMalloc(&labels_vec, sizeof(double) * N));
cudacall(cudaMemcpy(labels_vec, b, sizeof(double) * N , cudaMemcpyHostToDevice));	
cudacall(cudaMalloc(&fsvec, sizeof(double) * N));

threads_per_block = 512;
number_of_blocks = upit(N, threads_per_block);
forward_substitution_vector<<<1, 1>>>(M, labels_vec, fsvec, N);
cudaThreadSynchronize();

// Checking for forwardSubstitutionVector
check_forward_sub_vector<<<1, 1>>>(M, fsvec, labels_vec, N);
cudaThreadSynchronize();


// BACKWARD SUBSTITUTION

double *bsvec;
//	Allocating appropriate memory chunks
cudacall(cudaMalloc(&bsvec, sizeof(double) * N));
backward_substitution_vector<<<1, 1>>>(M, fsvec, bsvec, N); // Will use M.transpose() inside!!
cudaThreadSynchronize();

// Checking for backwardSubstitutionVector
check_backward_sub_vector<<<1, 1>>>(M, bsvec, fsvec, N);
cudaThreadSynchronize();
} */

void run_gp()
{

	double startime = CycleTimer::currentSeconds();
	double ans  = compute_log_likelihood();
	double endtime = CycleTimer::currentSeconds();
	printf("The time taken in loglikelihood computation = %lf\n", endtime - startime);

	Eigen::VectorXd params(3);

	double *localhp_grad = new double[3];
	compute_gradient_log_hyperparams(localhp_grad);
}

void test_matrix_mult()
{
	double *M1_host, *M2_host, *M3_host, *M3_host_noshare, *M3_host_share, *M1, *M2, *M3_noshare, *M3_share;
	int rows1, cols1, rows2, cols2, rows3, cols3;

	rows1 = 4096;
	cols1 = 4096;
	rows2 = 4096;
	cols2 = 4096;
	rows3 = 4096;
	cols3 = 4096;

	M1_host = new double[rows1 * cols1];
	M2_host = new double[rows2 * cols2];
	M3_host = new double[rows3 * cols3];
	M3_host_share = new double[rows3 * cols3];
	M3_host_noshare = new double[rows3 * cols3];

	cudacall(cudaMalloc(&M1, sizeof(double) * rows1 * cols1));
	cudacall(cudaMalloc(&M2, sizeof(double) * rows2 * cols2));
	cudacall(cudaMalloc(&M3_share, sizeof(double) * rows3 * cols3));
	cudacall(cudaMalloc(&M3_noshare, sizeof(double) * rows3 * cols3));

	int setter = 1;
	for (int i = 0; i < rows1 * cols1; i++)
	{
		M1_host[i] = rand() % 10;
		setter++;
	}

	srand(time(NULL));
	for (int i = 0; i < rows2 * cols2; i++)
	{
		M2_host[i] = rand() % 10;
		setter++;
	}

	double startime1 = CycleTimer::currentSeconds();	
	for (int i = 0; i < rows1; i++)
	{
		for (int j = 0; j < cols2; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < cols1; k++)
				sum += M1_host[i * cols1 + k] * M2_host[k * cols2 + j];
			M3_host[i * cols3 + j] = sum;
		}
	}
	printf("M3 host done\n");
	double endtime1 = CycleTimer::currentSeconds();

	/* printf("Matrix1:\n");
	for (int i = 0; i < rows1; i++)
	{
		for (int j = 0; j < cols1; j++)
			printf("%lf ", M1_host[i * cols1 + j]);
		printf("\n");
	}

	printf("Matrix2:\n");
	for (int i = 0; i < rows2; i++)
	{
		for (int j = 0; j < cols2; j++)
			printf("%lf ", M2_host[i * cols2 + j]);
		printf("\n");
	}
	
	printf("Matrix3:\n");
	for (int i = 0; i < rows3; i++)
	{
		for (int j = 0; j < cols3; j++)
			printf("%lf ", M3_host[i * cols3 + j]);
		printf("\n");
	} */

	cudacall(cudaMemcpy(M1, M1_host, sizeof(double) * rows1 * cols1, cudaMemcpyHostToDevice));	
	cudacall(cudaMemcpy(M2, M2_host, sizeof(double) * rows2 * cols2, cudaMemcpyHostToDevice));	

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(upit(cols2, blockDim.x), upit(rows1, blockDim.y));

	double startime2 = CycleTimer::currentSeconds();	
	matrixmultiply_noshare<<<gridDim, blockDim>>>(M1, rows1, cols1, M2, rows2, cols2, M3_noshare);
	cudaThreadSynchronize();
	double endtime2 = CycleTimer::currentSeconds();
	printf("No share done\n");

	cudacall(cudaMemcpy(M3_host_noshare, M3_noshare, sizeof(double) * rows3 * cols3, cudaMemcpyDeviceToHost));	
	cudaThreadSynchronize();

	/*printf("Matrix3 No share:\n");
	for (int i = 0; i < rows3; i++)
	{
		for (int j = 0; j < cols3; j++)
			printf("%lf ", M3_host_noshare[i * cols3 + j]);
		printf("\n");
	}*/

	dim3 blockDim2(BLOCK_SIZE, BLOCK_SIZE);
    
        dim3 gridDim2((cols2 + blockDim2.x - 1) / blockDim2.x, 
	                 (rows1 + blockDim2.y - 1) / blockDim2.y);

	double startime3 = CycleTimer::currentSeconds();	
	kernelSharedMemMatMult<<<gridDim2, blockDim2>>>(M1, rows1, cols1, M2, rows2, cols2, M3_share);
	cudaThreadSynchronize();
	double endtime3 = CycleTimer::currentSeconds();	

	cudacall(cudaMemcpy(M3_host_share, M3_share, sizeof(double) * rows3 * cols3, cudaMemcpyDeviceToHost));	
	cudaThreadSynchronize();

	printf("Share done\n");
	printf("Matrix3 Share:\n");
	/* for (int i = 0; i < rows3; i++)
	{
		for (int j = 0; j < cols3; j++)
			printf("%lf ", M3_host_share[i * cols3 + j]);
		printf("\n");
	} */

	for (int i = 0; i < rows3; i++)
	{
		for (int j = 0; j < cols3; j++)
			if (M3_host[i * cols3 + j] != M3_host_noshare[i * cols3 + j])
				correct = false;
	}

	printf("%s\n", correct?"CORRECT":"FALSE");
	correct = true;
	for (int i = 0; i < rows3; i++)
	{
		for (int j = 0; j < cols3; j++)
		//	if (M3_host_share[i * cols3 + j] != M3_host[i * cols3 + j] || M3_host_noshare[i * cols3 + j] != M3_host[i * cols3 + j])
			if (M3_host[i * cols3 + j] != M3_host_share[i * cols3 + j])
				correct = false;
	}

	printf("%s\n", correct?"CORRECT":"FALSE");
	printf("CPU: %lf\nGPU NO SHARE: %lf\nGPU SHARE: %lf\n", endtime1 - startime1, endtime2 - startime2, endtime3 - startime3);
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

void get_inverse_by_tmi(double *lower_triangular_mat, int ltm_dim)
{
	int mat_size, i, num_iters;
	int total_threads;
	double *final_ans;
	double *lower_triangular_mat_host;
	lower_triangular_mat_host = new double[ltm_dim * ltm_dim];

	int threads_per_block = 1024;
	int num_blocks = upit(ltm_dim / 2, threads_per_block);

	cudacall(cudaMemcpy(lower_triangular_mat_host, lower_triangular_mat, sizeof(double) * ltm_dim * ltm_dim, cudaMemcpyDeviceToHost));	

	inplace_lower_inverse_2x2<<<num_blocks, threads_per_block>>>(lower_triangular_mat, ltm_dim);
	cudaThreadSynchronize();

	num_iters = log2((double)ltm_dim) - 1;
	printf("num_iters is %d\n", num_iters);

	mat_size = 2;
	double startime, endtime;
	final_ans = new double[ltm_dim * ltm_dim];
	for (i = 0; i < num_iters; i++)
	{
		startime = CycleTimer::currentSeconds();	
		total_threads = ltm_dim * mat_size / 2;
		printf("Total threads launched: %d\n", total_threads);

		threads_per_block = 1024;
		num_blocks = upit(total_threads, threads_per_block);

		first_offseted_mat_mult<<<num_blocks, threads_per_block>>>(lower_triangular_mat, mat_size, tmi_intermediate_output, ltm_dim, total_threads);
		cudaThreadSynchronize();

		second_offseted_mat_mult<<<num_blocks, threads_per_block>>>(lower_triangular_mat, mat_size, tmi_intermediate_output, ltm_dim, total_threads);
		cudaThreadSynchronize();
		endtime = CycleTimer::currentSeconds();

		mat_size *= 2;
		printf("Time for iter %d: %lf\n", i, endtime - startime);
	}

	cudacall(cudaMemcpy(final_ans, lower_triangular_mat, sizeof(double) * ltm_dim * ltm_dim, cudaMemcpyDeviceToHost));	

	double total_sum = 0;
	for (int i = 0; i < ltm_dim; i++)
	{
		for (int j = 0; j < ltm_dim; j++)
		{
			double sum = 0;
			for (int k = 0; k < ltm_dim; k++)
				sum += lower_triangular_mat_host[i * ltm_dim + k] * final_ans[k * ltm_dim + j];
			//printf("%lf ", sum);
			total_sum += sum;
		}
		//printf("\n");
	}
	printf("Total sum: %lf\n", total_sum);

	delete []lower_triangular_mat_host;
	delete []final_ans;
}

 void test_tmi() {
	int ltm_dim = 2048;
	double *lower_triangular_mat_host;
	double *final_ans;
	int filler = 1;
	int i, j;
	int num_iters;
	int mat_size;
	int total_threads;

	lower_triangular_mat_host = new double[ltm_dim * ltm_dim];

	for (i = 0; i < ltm_dim; i++)
		for (j = 0; j <= i; j++)
			lower_triangular_mat_host[i * ltm_dim + j] = filler++;

	cudacall(cudaMalloc(&lower_triangular_mat, sizeof(double) * ltm_dim * ltm_dim));
	cudacall(cudaMemcpy(lower_triangular_mat, lower_triangular_mat_host, sizeof(double) * ltm_dim * ltm_dim, cudaMemcpyHostToDevice));	

	cudacall(cudaMalloc(&tmi_playground, sizeof(double) * ltm_dim * ltm_dim));

	final_ans = new double[ltm_dim * ltm_dim];
	double startime, endtime;

	int threads_per_block = 1024;
	int num_blocks = upit(ltm_dim / 2, threads_per_block);
	startime = CycleTimer::currentSeconds();	
	inplace_lower_inverse_2x2<<<num_blocks, threads_per_block>>>(lower_triangular_mat, ltm_dim);
	cudaThreadSynchronize();

	num_iters = log2((double)ltm_dim) - 1;
	printf("num_iters is %d\n", num_iters);

	mat_size = 2;
	for (i = 0; i < num_iters; i++)
	{
		total_threads = ltm_dim * mat_size / 2;
		printf("Total threads launched: %d\n", total_threads);

		threads_per_block = 1024;
		num_blocks = upit(total_threads, threads_per_block);

		first_offseted_mat_mult<<<num_blocks, threads_per_block>>>(lower_triangular_mat, mat_size, tmi_playground, ltm_dim, total_threads);
		cudaThreadSynchronize();

		second_offseted_mat_mult<<<num_blocks, threads_per_block>>>(lower_triangular_mat, mat_size, tmi_playground, ltm_dim, total_threads);
		cudaThreadSynchronize();

		mat_size *= 2;
	}

	endtime = CycleTimer::currentSeconds();
	printf("Total time for LMI = %lf\n",  endtime - startime);
	//printf("Final matrix:\n");
	// print_matrix_kernel<<<1, 1>>>(lower_triangular_mat, ltm_dim, ltm_dim);
	// cudaThreadSynchronize();

	cudacall(cudaMemcpy(final_ans, lower_triangular_mat, sizeof(double) * ltm_dim * ltm_dim, cudaMemcpyDeviceToHost));	

	double total_sum = 0;
	for (int i = 0; i < ltm_dim; i++)
	{
		for (int j = 0; j < ltm_dim; j++)
		{
			double sum = 0;
			for (int k = 0; k < ltm_dim; k++)
				sum += lower_triangular_mat_host[i * ltm_dim + k] * final_ans[k * ltm_dim + j];
			//printf("%lf ", sum);
			total_sum += sum;
		}
		//printf("\n");
	}
	printf("Total sum: %lf\n", total_sum);
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

//compute_test_means_and_variances is a set of kernels
void compute_test_means_and_variances(){
	int threads_per_block, number_of_blocks;

	//Maybe can move the compute_K_train to setup in SCHEDULER-vala (THINK ABOUT IT SID)
        threads_per_block = 512;
        number_of_blocks = upit((N * N), threads_per_block);
        compute_K_train<<<number_of_blocks, threads_per_block>>>(X, K, loghyper, N, DIM); // populated in K
        cudaThreadSynchronize();
	
	//compute_K_inverse(); //populates Kinv with K.inverse()
	// instead of compute_K_inverse, let's see if TMI is of help!!!
	compute_K_inverse_with_tmi();	
	
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

		vector_dot_product<<<1, 1>>>(Ktest_vec, temp1dvec, tmeanvec + i, N); //for mean
      		cudaThreadSynchronize();
	
		threads_per_block = 512;
	        number_of_blocks = upit(N, threads_per_block);
       		vector_matrix_multiply<<<number_of_blocks, threads_per_block>>>(Ktest_vec, Kinv, temp_fs, N); //REUSING temp_fs from likelihood computation
        	cudaThreadSynchronize();
		
		vector_dot_product_with_loghp<<<1, 1>>>(Ktest_vec, temp_fs, tvarvec + i, N, sig_var, noise_var ); //for variance
      		cudaThreadSynchronize();
		
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
	compute_test_means_and_variances();

		
	printf("\n---------------------------------------\n");
	printf("Now result time\n");	
	printf("---------------------------------------\n");
	// actual answer time
	get_negative_log_predprob();
	
}
