#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void
hardcoded_cholesky_2x2()
{
	printf("In kernel\n");
}

void run_kernel()
{
	printf("Call kernel\n");
	hardcoded_cholesky_2x2<<<1,1>>>();
	cudaThreadSynchronize();
	printf("Kernel call done\n");
}
