#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

double *M;
double *a11;

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

	a11[0] = M[start_id * dim + start_id] = sqrt(M[start_id * dim + start_id]);
	a11[1] = M[start_id * dim + start_id + 1] = 0.0;
	a11[2] = M[(start_id + 1) * dim + start_id] = M[(start_id + 1) * dim + start_id] / M[start_id * dim + start_id];
	a11[3] = M[(start_id + 1) * dim + start_id + 1] = sqrt(M[(start_id + 1) * dim + start_id + 1] - M[(start_id + 1) * dim + start_id] * M[(start_id + 1) * dim + start_id]);

	printf("printing a11 matrix\n");
	printf("%lf %lf %lf %lf\n", a11[0], a11[1], a11[2], a11[3]);
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

	/*
	 * First generate the M matrix
	 */
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

	/*
	 * Now malloc the a11 matrix
	 */

	cudacall(cudaMalloc(&a11, sizeof(double) * 4));

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


void run_kernel()
{
	int dim, start_id, b;

	start_id = 0;
	dim = 8;
	b = 2;
	start_id = 0;

	setup(dim);

	for (int i = 0; i < dim / b; i++)
	{
		hardcoded_cholesky_2x2<<<1, 1>>>(M, a11, dim, b, start_id);
		cudaThreadSynchronize();
		start_id += b;
	}
	printf("Kernel call done\n");
}
