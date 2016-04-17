#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdlib>
#include "cycleTimer.h"
#include <fstream>

#define filename "sym5000_1.txt"

double *temp_m; 
double *orig_sym; //orig matrix for reference
 
double *mt;
double *mt_transpose;

double *M;
double *a11;
double *a21_transpose;
double *l21_transpose_from_fs;
double *l21;
double *l22_temp;  //This is for updating a22

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
			printf("%lf ", arr[i * dim2 + j]);
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

	long long int col = blockIdx.y * blockDim.y + threadIdx.y;
	long long int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= rowsA || col >= colsB)
		return;

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

void setup(int dim, int b)
{
	

	cudacall(cudaMalloc(&M, sizeof(double) * dim * dim));
	cudacall(cudaMalloc(&mt, sizeof(double)*dim * dim));
	cudacall(cudaMalloc(&mt_transpose, sizeof(double)*dim * dim));
	cudacall(cudaMemcpy(mt, temp_m, sizeof(double) * dim * dim, cudaMemcpyHostToDevice));

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

__inline__ int upit(int x, int y) {
	return (x + y - 1) / y;
}


void check_cholesky(double *M1, double* targetoutput, int d){
	double diff = 0.0, totaldiff = 0.0;	
	for(int i = 0; i < d; i++){
		for(int j = 0; j < d ;j++){ 
			double tempval = 0.0;
			for(int k = 0; k < d; k++){
				//tempval += M1[i*d + k] * M2[k * d + j];
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

void run_kernel()
{
	int dim, start_id, b;
	int threads_per_block;
	int number_of_blocks;
	int num_iters;

	double startime, endtime ;
	start_id = 0;
	dim = 4500;
	b = 2;
	start_id = 0;

	init_and_print();
//	get_input(dim);
	initialize_random(dim);
	printf("okay random bhar gaya\n");
	setup(dim, b);
	
	// Input generation
	//	1. taking transpose of mt in mt_transpose
	threads_per_block = 256;
	number_of_blocks = upit(dim * dim, threads_per_block);
	generic_matrix_transpose<<<number_of_blocks, threads_per_block>>>(mt, mt_transpose, dim, dim);
	cudaThreadSynchronize();
	printf("ab jakar transpose hua\n");

	/*
	print_matrix_kernel<<<1, 1>>>(mt, dim, dim);
	cudaThreadSynchronize();
	print_matrix_kernel<<<1, 1>>>(mt_transpose, dim, dim);
	cudaThreadSynchronize();
	*/
	
	startime = CycleTimer::currentSeconds();	
	dim3 blockDimTemp(32,32);
	dim3 gridDimTemp( upit(dim, blockDimTemp.x), upit(dim, blockDimTemp.y));
	//matrixmultiply_noshare(double *a, int rowsA, int colsA, double *b, int rowsB, int colsB, double *c)
	matrixmultiply_noshare<<<gridDimTemp, blockDimTemp >>>(mt, dim, dim,  mt_transpose, dim, dim, M);
	cudaThreadSynchronize();
	endtime = CycleTimer::currentSeconds();	
	printf("Now multiplication got over, total time taken for dim = %d, is %lf\n", dim, endtime - startime);

	// Now copying the symmetric matrix from CUDA to host
	orig_sym = new double[dim * dim];
	cudacall(cudaMemcpy(orig_sym, M,  sizeof(double) * dim * dim, cudaMemcpyDeviceToHost));
	
	printf("Host me aya kyaa??\n");
	std::ofstream out(filename);
	for(int i = 0; i < dim ; i++){
		for(int j = 0; j < dim ; j++){
			out << orig_sym[i*dim + j] << " ";
	//		printf("%lf ", orig_sym[i*dim + j]);
		}
		out << "\n";
	//	printf("\n");
	}
	
	out.close();
	return;
	
	startime = CycleTimer::currentSeconds();
	num_iters = dim / b;
	for (int i = 0; i < num_iters; i++)
	{
		hardcoded_cholesky_2x2<<<1, 1>>>(M, a11, dim, b, start_id);
		cudaThreadSynchronize();

		if (i == num_iters - 1)
			break;

		// TODO optimize a21_transpose, by bypassing it perhaps? Can avoid transpose and manipulate indices inside next kernel
		threads_per_block = 256;
		number_of_blocks = upit((dim - b - start_id) * b, threads_per_block);
		take_a21_transpose<<<number_of_blocks, threads_per_block>>>(M, a21_transpose, dim, b, start_id);
		cudaThreadSynchronize();

		threads_per_block = 256;
		number_of_blocks = upit((dim - b - start_id), threads_per_block);
		forward_substitution_rectangular_a21<<<number_of_blocks, threads_per_block>>>(M, a11, a21_transpose, l21_transpose_from_fs, dim, b, start_id);
		cudaThreadSynchronize();

	//	printf("Printing l21_transpose_from_fs\n");
	//	print_matrix_kernel<<<1, 1>>>(l21_transpose_from_fs, b, dim - b - start_id);
	//	cudaThreadSynchronize();

		/*		
		printf("\n\n");
		printf(" ---------------------------------------- \n");	
		print_matrix_kernel<<<1, 1>>>(a11, b, b);
		cudaThreadSynchronize();
		printf(" ---------------------------------------- \n");
		print_matrix_kernel<<<1,1>>>(a21_transpose, b, dim - b - start_id);
		cudaThreadSynchronize();
		printf(" ---------------------------------------- \n");
		singlethread_temp_matmult_kernel<<<1, 1>>>(a11, a21_transpose, l21_transpose_from_fs, b, b, dim - b - start_id);
		cudaThreadSynchronize();
		print_matrix_kernel<<<1,1>>>(l21_transpose_from_fs, b, dim - b - start_id);
		cudaThreadSynchronize();
		printf("\n\n");
		*/
			
		//printf("\nNow printing entire M matrix\n");
		//print_matrix_kernel<<<1, 1>>>(M, dim, dim);
		//cudaThreadSynchronize();
		
		// TODO: Can include this tranpose in the forward_substitution_rectangular_a22 call!!!!
		// Now taking transpose of l21_transpose_from_fs
		 
		threads_per_block = 256;
		number_of_blocks = upit((dim - b - start_id) * b, threads_per_block);
		generic_matrix_transpose<<<number_of_blocks, threads_per_block>>>(l21_transpose_from_fs, l21, b, dim - b - start_id);
		cudaThreadSynchronize();
		
//		printf("\nNow checking the transpose => \n");	
//		print_matrix_kernel<<<1,1>>>(l21, dim - b - start_id, b);
//		cudaThreadSynchronize();
//		printf("Checking the l21_transpose_from_fs matrix\n");
//		check_l21_kernel<<<1, 1>>>(a11, l21_transpose_from_fs, a21_transpose, b, b, dim - b - start_id);
//		cudaThreadSynchronize();

		//matrixmultiply_noshare<<<(double *a, int rowsA, int colsA, double *b, int rowsB, int colsB, double *c)
		int rowA = (dim - b - start_id) , colA = b, rowB = b , colB = (dim - b - start_id) ;
		dim3 blockDim(2,2);
		dim3 gridDim( upit(colB, blockDim.x), upit(rowA, blockDim.y));
		matrixmultiply_noshare<<<gridDim, blockDim >>>(l21, (dim - b - start_id), b,  l21_transpose_from_fs, b, dim - b - start_id, l22_temp);
		cudaThreadSynchronize();

		threads_per_block = 256;
		number_of_blocks = upit((dim - b - start_id) * (dim - b - start_id), threads_per_block);
		offseted_elementwise_subtraction<<<number_of_blocks, threads_per_block >>>(l22_temp, dim - b - start_id, M, dim, b, start_id);
		cudaThreadSynchronize();

		start_id += b;
	}
	// Fire a kernel for making upper-triangular as 0.0
	threads_per_block = 256;
	number_of_blocks = upit( (dim * dim), threads_per_block);
	set_upper_zero<<<number_of_blocks, threads_per_block>>>(M, dim);
	cudaThreadSynchronize();
	endtime = CycleTimer::currentSeconds();	
	printf("Totat time taken = %lf s\n", endtime - startime);	
	// Now checking!
	
	double *finalans = new double[dim * dim];
	cudacall(cudaMemcpy(finalans, M,  sizeof(double) * dim * dim, cudaMemcpyDeviceToHost));
	check_cholesky(finalans, orig_sym, dim);	
	
	/*for(int i = 0; i < dim ; i++){
		for(int j = 0; j < dim ; j++){
			printf("%lf ", finalans[i*dim + j]);
		}
		printf("\n");
	}*/
	
}
