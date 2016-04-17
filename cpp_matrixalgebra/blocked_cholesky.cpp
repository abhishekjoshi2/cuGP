#include<iostream>
#include<cmath>
#include <cstdlib>
#include "../common/matrixops.h"
#include <cstdio>
#include "../common/cycleTimer.h"

void matrix_forward_substitution_rectangular(double **, double **, double **temp_output, int b, int dim);

void matrix_transpose(double **, double**, int);

void get_symmetric_matrix(double **M, double **matrix1, double **matrix2, int dim){

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
                                M[i][j] += matrix1[i][k]*matrix2[k][j];
                        }
                }
        } 
	/*int setter = 50;
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
		{
			M[i][j] = setter++;
			M[j][i] = M[i][j];
		}*/
}

void check_matrixans_rect(double **arr1, double **arr2, double **arr3, int dim1, int dim2)
{
	printf("Checking correctness for rect matrix\n");
	int i, j, k;
	double total_sum = 0.0;
	for(i = 0; i < dim1; i++) {
		for(j = 0; j < dim2; j++) {
			double temp = 0.0;
			for(k = 0; k < dim1; k++) {
				temp += arr1[i][k] * arr2[k][j];
			}
			// printf("%lf - %lf = %lf\n", temp, arr3[i][j], temp - arr3[i][j]);
			total_sum += (temp - arr3[i][j]) * (temp - arr3[i][j]);
		}
	}
	printf("Final error is %lf\n", total_sum);
}

// this is a special type of transpose for the A21 matrix, since it is within the actual M matrix
void get_offseted_transpose(double **M, int start_idx, int dim, int b, double **temp_a21) {
	int rows, cols;

	rows = dim - start_idx - b;
	cols = b;

	//printf("In offseted_transpose, have to take transpose of %dx%d matrix\n", rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			temp_a21[j][i] = M[start_idx + b + i][start_idx + j];
		}
	}
}

void hardcoded_cholesky_inplace_and_a11(double **M , int idx, int dim, int b, double **a11) {
	//printf("In hardcoded cholesky, idx: %d, dim: %d, b: %d\n", idx, dim, b);
	if (b == 1) {
		M[idx][idx] = sqrt(M[idx][idx]);
		a11[0][0] = M[idx][idx];
	}	
	else if (b == 2) {
		a11[0][0] = M[idx][idx] = sqrt(M[idx][idx]);
		a11[0][1] = M[idx][idx + 1] = 0.0;
		a11[1][0] = M[idx + 1][idx] = M[idx + 1][idx] / M[idx][idx];
		a11[1][1] = M[idx + 1][idx + 1] = sqrt(M[idx + 1][idx + 1] - M[idx + 1][idx] * M[idx + 1][idx]);

		/*idx = 0;
		a11[idx][idx] = M[idx][idx];
		a11[idx][idx + 1] = M[idx][idx + 1];
		a11[idx + 1][idx] = M[idx + 1][idx];
		a11[idx + 1][idx + 1] = M[idx + 1][idx + 1];
		printf("A11 is:\n");*/
		print_matrix(a11, b, b);
	}
}

void update_A21_temp_and_inplace(double **M, double**temp_a21_transpose, double **temp_output, double **a11, int start_idx, int dim, int b) {
	// first make changes in temp_a21_transpose, and then copy those back into M
	// temp_a21_transpose can potentially be a small matrix, but for simplicity, we make it a dimxdim matrix before passing

	//printf("In update A21, start_idx: %d, dim: %d, b: %d\n", start_idx, dim, b);

	// first get offsetted transpose
	get_offseted_transpose(M, start_idx, dim, b, temp_a21_transpose);
	// we now have the a21_transpose in temp_a21_transpose

	matrix_forward_substitution_rectangular(a11, temp_a21_transpose, temp_output, b, dim - b - start_idx);

	// now, temp_output has a bx(n-b) matrix, which should be copied back to M
	for (int i = 0; i < b; i++)
	{
		for (int j = 0; j < dim - start_idx - b; j++)
		{
			M[start_idx + j + b][start_idx + i] = temp_output[i][j];
			// adding this later XXX
			temp_a21_transpose[i][j] = temp_output[i][j];
		}
	}
}

void matrix_transpose_rectangular(double **input, double **a21, int rows, int cols)
{
	int i, j;
	//printf("In matrix transpose rectangular with rows: %d, cols: %d\n", rows, cols);

	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
			a21[j][i] = input[i][j];
}

void matrix_multiply_for_a22(double **mat1, double **mat2, double **product, int rows, int cols)
{
	double ans;

	//printf("In matrix multiply, rows is %d, cols is %d\n", rows, cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			ans = 0.0;
			for (int k = 0; k < cols; k++)
				ans += mat1[i][k] * mat2[k][j];
			product[i][j] = ans;
		}
	}
}

void a22_update(double **M, double **a21, double **a21_transpose, int start_idx, int dim, int b)
{
	//printf("In a22_update, start_idx is %d, dim is %d, b is %d\n", start_idx, dim, b);
	double **temp_product;
	int new_dim = dim - b - start_idx;

	temp_product = new double*[new_dim];

	for (int i = 0; i < new_dim; i++)
		temp_product[i] = new double[new_dim];

	//printf("a21: \n");
	//print_matrix(a21, new_dim, b);
	//printf("a21_transpose:\n");
	//print_matrix(a21_transpose, b, new_dim);
	matrix_multiply_for_a22(a21, a21_transpose, temp_product, new_dim, b);
	//printf("temp_product is: \n");

	//print_matrix(temp_product, new_dim, new_dim);

	for (int i = 0; i < new_dim; i++)
		for (int j = 0; j < new_dim; j++)
			M[start_idx + b + i][start_idx + b + j] = M[start_idx + b + i][start_idx + b + j] - temp_product[i][j];
}

int main() {
	
	int dim = 1000;
	double **M, **matrix1, **matrix2;
	double **M_orig, **M_trans;
	double **temp_a21_transpose, **temp_output;
	double **a11;
	double **a21;
	temp_a21_transpose = new double *[dim];
	temp_output = new double *[dim];

	int b = 2;
	
	a11 = new double *[b];
	for (int i = 0; i < b; i++)
	{
		a11[i] = new double[b];
	}

	a21 = new double *[dim];
	M = new double*[dim];	
	matrix1 = new double*[dim];	
	matrix2 = new double*[dim];	
	M_orig = new double*[dim];
	M_trans = new double*[dim];
	for (int i = 0; i < dim; i++) {
		M[i] = new double[dim];
		temp_a21_transpose[i] = new double[dim];
		temp_output[i] = new double[dim];
		a21[i] = new double[dim];
		matrix1[i] = new double[dim];
		matrix2[i] = new double[dim];
		M_orig[i] = new double[dim];
		M_trans[i] = new double[dim];
	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			M[i][j] = 0;
		
	get_symmetric_matrix(M, matrix1, matrix2, dim);

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			M_orig[i][j] = M[i][j];

	//printf("Got matrix as \n");
	//print_matrix(M, dim, dim);

	double startime = CycleTimer::currentSeconds();
	int iter = dim / b;
	int start_idx = 0;
	for (int i = 0; i < iter; i++) {
		//printf("Doing hardcoded cholesky\n");
		hardcoded_cholesky_inplace_and_a11(M, start_idx, dim, b, a11);
		//printf("Hardcoded cholesky done\n");
		//printf("Matrix is: \n");
		//print_matrix(M, dim, dim);

		//printf("Trying to update A21\n");
		update_A21_temp_and_inplace(M, temp_a21_transpose, temp_output, a11, start_idx, dim, b);
		//printf("Updating A21 done\n");
		//printf("Matrix is: \n");
		//print_matrix(M, dim, dim);

		//printf("Trying to get matrix transpose rectangular\n");
		matrix_transpose_rectangular(temp_a21_transpose, a21, b, dim - start_idx - b);
		//printf("Get matrix transpose rectangular\n");
		//printf("Matrix is: \n");
		//print_matrix(M, dim, dim);
		
		//printf("Trying to update a22\n");
		a22_update(M, a21, temp_a21_transpose, start_idx, dim, b);
		//printf("Update A22 done\n");
		//printf("Matrix is: \n");
		//print_matrix(M, dim, dim);

		//printf("After round %d, matrix is:\n", i);
		//print_matrix(M, dim, dim);
		start_idx += b;
		//printf("\n\n");
	}

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			if (j > i)
				M[i][j] = 0.0;
		}
	}

	double endtime = CycleTimer::currentSeconds();
	
	//printf("Final matrix is:\n");
	//print_matrix(M, dim, dim);
	printf("Totat time taken = %lf s\n", endtime - startime);
	matrix_transpose(M, M_trans, dim);

	check_matrixans_rect(M, M_trans, M_orig, dim, dim);
	
	return 0;
}
