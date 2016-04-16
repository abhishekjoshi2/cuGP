#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

double *M;

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


	__global__ void
hardcoded_cholesky_2x2(double *M, int dim, int start_id)
{
	int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
	int j_index = (blockIdx.y * blockDim.y + threadIdx.y);

	printf("In kernel\n");
	printf("dim is %d, i_index is %d, j_index is %d\n", dim, i_index, j_index);

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			printf("%lf ", M[i * dim + j]);
		}
		printf("\n");
	}
}


void get_symmetric_matrix_1d(double *M, double **matrix1, double **matrix2, int dim) {

	srand(time(NULL));
	for (int i = 0; i < dim; i++){
		for (int j = 0; j < dim; j++){
			matrix1[i][j] = rand() % 100 + 1;
			matrix2[j][i] = matrix1[i][j];
		}
	}

	for (int i = 0; i < dim; i++){
		for(int j = 0; j < dim; j++){
			for(int k = 0; k < dim; k++){
				M[i * dim + j] += matrix1[i][k]*matrix2[k][j];
			}
		}
	}
}

void setup(int dim)
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

	// By this time the scene should be loaded.  Now copy all the key
	// data structures into device memory so they are accessible to
	// CUDA kernels
	//
	// See the CUDA Programmer's Guide for descriptions of
	// cudaMalloc and cudaMemcpy

	double *temp_m, **m1, **m2;

	temp_m = new double[dim * dim];

	m1 = new double *[dim];
	m2 = new double *[dim];

	for (int i = 0; i < dim; i++)
	{
		m1[i] = new double[dim];
		m2[i] = new double[dim];
	}

	get_symmetric_matrix_1d(temp_m, m1, m2, dim);

	printf("Generated matrix in host is \n");
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			printf("%lf ", temp_m[i * dim + j]);
		}
		printf("\n");
	}

	cudacall(cudaMalloc(&M, sizeof(double) * dim * dim));

	cudacall(cudaMemcpy(M, temp_m, sizeof(double) * dim * dim, cudaMemcpyHostToDevice));

	// Initialize parameters in constant memory.  We didn't talk about
	// constant memory in class, but the use of read-only constant
	// memory here is an optimization over just sticking these values
	// in device global memory.  NVIDIA GPUs have a few special tricks
	// for optimizing access to constant memory.  Using global memory
	// here would have worked just as well.  See the Programmer's
	// Guide for more information about constant memory.

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

	// last, copy over the color table that's used by the shading
	// function for circles in the snowflake demo

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


void run_kernel()
{
	int dim = 8, start_id;
	setup(dim);
	printf("Call kernel\n");
	// hardcoded_cholesky_2x2<<<1,1>>>();
	
	start_id = 0;

	hardcoded_cholesky_2x2<<<1, 1>>>(M, dim, start_id);
	cudaThreadSynchronize();
	printf("Kernel call done\n");
}
