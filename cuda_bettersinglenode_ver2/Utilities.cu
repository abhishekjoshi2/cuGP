#include <stdio.h>
#include <assert.h>
//#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

#if __CUDA_ARCH__ >= 700
#include <cusolverDn.h>
#endif
#include <cublas_v2.h>
#include <cufft.h>

#include "Utilities.cuh"

#define DEBUG

/*******************/
/* iDivUp FUNCTION */
/*******************/
extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  if (abort) { exit(code); }
   }
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**************************/
/* CUSOLVE ERROR CHECKING */
/**************************/
#if __CUDA_ARCH__ >= 700
static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if(CUSOLVER_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSOLVE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs",__FILE__, __LINE__,err, \
                                _cusolverGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }
#endif

/*************************/
/* CUBLAS ERROR CHECKING */
/*************************/
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
}

    return "<unknown>";
}

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if(CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs",__FILE__, __LINE__,err, \
                                _cublasGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cublasSafeCall(cublasStatus_t err) { __cublasSafeCall(err, __FILE__, __LINE__); }

/************************/
/* CUFFT ERROR CHECKING */
/************************/
static const char *_cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

// --- CUFFTSAFECALL
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n \nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, _cufftGetErrorEnum(err));
		cudaDeviceReset(); assert(0);
    }
}

extern "C" void cufftSafeCall(cufftResult err) { __cufftSafeCall(err, __FILE__, __LINE__); }

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {

		case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

    return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if(CUSPARSE_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs",__FILE__, __LINE__,err, \
                                _cusparseGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

/************************/
/* REVERSE ARRAY KERNEL */
/************************/
#define BLOCKSIZE_REVERSE	256

// --- Credit to http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/208801731?pgno=2
template <class T>
__global__ void reverseArrayKernel(const T * __restrict__ d_in, T * __restrict__ d_out, const int N, const T a)
{
	// --- Credit to the simpleTemplates CUDA sample
	SharedMemory<T> smem;
    T* s_data = smem.getPointer();

    const int tid			= blockDim.x * blockIdx.x + threadIdx.x;
	const int id			= threadIdx.x;
	const int offset		= blockDim.x * (blockIdx.x + 1);

	// --- Load one element per thread from device memory and store it *in reversed order* into shared memory
	if (tid < N) s_data[BLOCKSIZE_REVERSE - (id + 1)] = a * d_in[tid];

	// --- Block until all threads in the block have written their data to shared memory
	__syncthreads();

	// --- Write the data from shared memory in forward order
	if ((N - offset + id) >= 0) d_out[N - offset + id] = s_data[threadIdx.x];
}

/************************/
/* REVERSE ARRAY KERNEL */
/************************/
template <class T>
void reverseArray(const T * __restrict__ d_in, T * __restrict__ d_out, const int N, const T a) {

    reverseArrayKernel<<<iDivUp(N, BLOCKSIZE_REVERSE), BLOCKSIZE_REVERSE, BLOCKSIZE_REVERSE * sizeof(T)>>>(d_in, d_out, N, a);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

template void reverseArray<float>  (const float  * __restrict__, float  * __restrict__, const int, const float);
template void reverseArray<double> (const double * __restrict__, double * __restrict__, const int, const double);

/********************************************************/
/* CARTESIAN TO POLAR COORDINATES TRANSFORMATION KERNEL */
/********************************************************/
#define BLOCKSIZE_CART2POL	256

template <class T>
__global__ void Cartesian2PolarKernel(const T * __restrict__ d_x, const T * __restrict__ d_y, T * __restrict__ d_rho, T * __restrict__ d_theta,
	                       const int N, const T a) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N) {
		d_rho[tid]		= a * hypot(d_x[tid], d_y[tid]);
		d_theta[tid]	= atan2(d_y[tid], d_x[tid]);
	}

}

/*******************************************************/
/* CARTESIAN TO POLAR COORDINATES TRANSFORMATION - GPU */
/*******************************************************/
//template <class T>
//thrust::pair<T *,T *> Cartesian2Polar(const T * __restrict__ d_x, const T * __restrict__ d_y, const int N, const T a) {
//
//	T *d_rho;	gpuErrchk(cudaMalloc((void**)&d_rho,   N * sizeof(T)));
//	T *d_theta; gpuErrchk(cudaMalloc((void**)&d_theta, N * sizeof(T)));
//
//	Cartesian2PolarKernel<<<iDivUp(N, BLOCKSIZE_CART2POL), BLOCKSIZE_CART2POL>>>(d_x, d_y, d_rho, d_theta, N, a);
//#ifdef DEBUG
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//#endif
//
//	return thrust::make_pair(d_rho, d_theta);
//}
//
//template thrust::pair<float  *, float  *>  Cartesian2Polar<float>  (const float  *, const float  *, const int, const float);
//template thrust::pair<double *, double *>  Cartesian2Polar<double> (const double *, const double *, const int, const double);

/*******************************************************/
/* CARTESIAN TO POLAR COORDINATES TRANSFORMATION - CPU */
/*******************************************************/
//template <class T>
//thrust::pair<T *,T *> h_Cartesian2Polar(const T * __restrict__ h_x, const T * __restrict__ h_y, const int N, const T a) {
//
//	T *h_rho	= (T *)malloc(N * sizeof(T));
//	T *h_theta	= (T *)malloc(N * sizeof(T));
//
//	for (int i = 0; i < N; i++) {
//		h_rho[i]	= a * hypot(h_x[i], h_y[i]);
//		h_theta[i]	= atan2(h_y[i], h_x[i]);
//	}
//
//	return thrust::make_pair(h_rho, h_theta);
//}
//
//template thrust::pair<float  *, float  *>  h_Cartesian2Polar<float>  (const float  *, const float  *, const int, const float);
//template thrust::pair<double *, double *>  h_Cartesian2Polar<double> (const double *, const double *, const int, const double);

/*******************************/
/* LINEAR COMBINATION FUNCTION */
/*******************************/
void linearCombination(const float * __restrict__ d_coeff, const float * __restrict__ d_basis_functions_real, float * __restrict__ d_linear_combination,
	                   const int N_basis_functions, const int N_sampling_points, const cublasHandle_t handle) {

    float alpha = 1.f;
    float beta  = 0.f;
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N_sampling_points, N_basis_functions, &alpha, d_basis_functions_real, N_sampling_points,
                               d_coeff, 1, &beta, d_linear_combination, 1));

}

void linearCombination(const double * __restrict__ d_coeff, const double * __restrict__ d_basis_functions_real, double * __restrict__ d_linear_combination,
	                   const int N_basis_functions, const int N_sampling_points, const cublasHandle_t handle) {

    double alpha = 1.;
    double beta  = 0.;
    cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, N_sampling_points, N_basis_functions, &alpha, d_basis_functions_real, N_sampling_points,
                               d_coeff, 1, &beta, d_linear_combination, 1));

}

/******************************/
/* ADD A CONSTANT TO A VECTOR */
/******************************/
#define BLOCKSIZE_VECTORADDCONSTANT	256

template<class T>
__global__ void vectorAddConstantKernel(T * __restrict__ d_in, const T scalar, const int N) {

	const int tid	= threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N) d_in[tid] += scalar;

}

template<class T>
void vectorAddConstant(T * __restrict__ d_in, const T scalar, const int N) {

	vectorAddConstantKernel<<<iDivUp(N, BLOCKSIZE_VECTORADDCONSTANT), BLOCKSIZE_VECTORADDCONSTANT>>>(d_in, scalar, N);

}

template void  vectorAddConstant<float> (float  * __restrict__, const float , const int);
template void  vectorAddConstant<double>(double * __restrict__, const double, const int);

/*****************************************/
/* MULTIPLY A VECTOR BY A CONSTANT - GPU */
/*****************************************/
#define BLOCKSIZE_VECTORMULCONSTANT	256

template<class T>
__global__ void vectorMulConstantKernel(T * __restrict__ d_in, const T scalar, const int N) {

	const int tid	= threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N) d_in[tid] *= scalar;

}

template<class T>
void vectorMulConstant(T * __restrict__ d_in, const T scalar, const int N) {

	vectorMulConstantKernel<<<iDivUp(N, BLOCKSIZE_VECTORMULCONSTANT), BLOCKSIZE_VECTORMULCONSTANT>>>(d_in, scalar, N);

}

template void  vectorMulConstant<float> (float  * __restrict__, const float , const int);
template void  vectorMulConstant<double>(double * __restrict__, const double, const int);

/*****************************************/
/* MULTIPLY A VECTOR BY A CONSTANT - CPU */
/*****************************************/
template<class T>
void h_vectorMulConstant(T * __restrict__ h_in, const T scalar, const int N) {

	for (int i = 0; i < N; i++) h_in[i] *= scalar;

}

template void  h_vectorMulConstant<float> (float  * __restrict__, const float , const int);
template void  h_vectorMulConstant<double>(double * __restrict__, const double, const int);

/*****************************************************/
/* FUSED MULTIPLY ADD OPERATIONS FOR HOST AND DEVICE */
/*****************************************************/
template<class T>
__host__ __device__ T fma2(T x, T y, T z) { return x * y + z; }

template float  fma2<float >(float , float , float );
template double fma2<double>(double, double, double);

/*******************/
/* MODULO FUNCTION */
/*******************/
__device__ int modulo(int val, int _mod)
{
	int P;
	if(val > 0) { (!(_mod & (_mod - 1))? P = val&(_mod-1) : P = val%(_mod)); return P; }
	else
	{
		(!(_mod & (_mod - 1))? P = (-val)&(_mod-1) : P = (-val)%(_mod));
		if(P > 0) return _mod -P;
		else return 0;
	}
}

/***************************************/
/* ATOMIC ADDITION FUNCTION ON DOUBLES */
/***************************************/
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    register unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/*********************************/
/* ATOMIC MIN FUNCTION ON FLOATS */
/*********************************/
__device__ float atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}


