#include<cstdio>

__global__ void set_only_one(double *M, int size){
        int i_index = (blockIdx.x * blockDim.x + threadIdx.x);
        int j_index = (blockIdx.y * blockDim.y + threadIdx.y);
	
        if(i_index != 0 || j_index != 2) return;
        M[j_index*size + i_index] = 1.0;
}

void caller(){
        int siz  = 4;
        dim3 blockDim(2,2);
        dim3 gridDim( 2,2 );
        double *arr1;
        cudaMalloc(&arr1, sizeof(double) * siz * siz);
        cudaMemset((void *)arr1, 0.0, sizeof(double) * siz * siz);
        set_only_one<<<gridDim, blockDim  >>>(arr1, siz);
        cudaThreadSynchronize();

        double *hostarr = new double[siz * siz];
        cudaMemcpy(hostarr, arr1,  sizeof(double)*siz * siz, cudaMemcpyDeviceToHost);
        for(int i = 0; i < siz; i++){
                for(int j = 0; j < siz;j++){
                        printf("%lf ", hostarr[i*siz + j]);
                }
                printf("\n");
        }

		
}
