#include<iostream>
#include<cmath>
#include<cstdlib>
#include "matrixops.h"

void get_symmetric_matrix(double **M, double **matrix1, double **matrix2, int dim){
	rand(time(NULL));
        for (int i = 0; i < dim; i++){
                for (int j = 0; j < dim; j++){
                        matrix1[i][j] = rand() % 100 + 1;
                        matrix2[j][i] = matrix1[i][j];
                }
        }

        for (int i = 0; i < dim; i++){
                for(int j = 0; j < dim; j++){
                        for(int k = 0; k < dim; k++){
                                M[i][j] += matrix1[i][k]*matrix2[k][j];
                        }
                }
        }
}

void hardcoded_cholesky(double **M , int idx, int dim, int b){
	if(b == 1){
		M[dx][idx] = sqrt(M[idx][idx]);
	}	
	else if(b == 2){
		M[idx][idx] = sqrt(M[idx][idx]);
		M[idx][idx + 1] = 0.0;
		M[idx + 1][idx] = M[idx + 1][idx] / M[idx][idx];
		M[idx + 1][idx + 1] = sqrt(M[idx + 1][idx + 1] - M[idx + 1][idx] * M[idx + 1][idx]);
	}
}

void updat
		update_A21(M, temp1, start_idx, dim, b);
int main(){
	
	int dim = 6;
	double **matrix1, **matrix2, **M;
	double **temp1, **temp2;
	matrix1 = new double*[dim];
	matrix2 = new double*[dim];
	M = new double*[dim];	
	temp1 = new double *[dim];
	temp2 = new double *[dim];
	
	for(int i = 0 ; i < dim ;i++){		
		matrix1[i] = new double[dim];
		matrix2[i] = new double[dim];
		M[i] = new double[dim];
		temp1[i] = new double[dim];
		temp2[i] = new double[dim];
	}
		
	get_symmetric_matrix(M, matrix1, matrix2, dim);

	int b = 2;
		
	int iter = dim / b;
	int start_idx = 0;
	for(int i = 0; i < iter; i++){
		hardcoded_cholesky(M, start_idx, dim, b);
		update_A21(M, temp1, start_idx, dim, b);
		right_matrixupdate(M, temp1, temp2, start_idx, dim, b);
		start_idx += b;
	}

	return 0;
}
