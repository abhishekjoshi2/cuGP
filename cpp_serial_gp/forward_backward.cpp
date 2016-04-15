#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>

#define DIM 10
void print_matrix(double m[][DIM], int rows, int cols) {
	for (int i = 0 ; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			printf("%lf\t", m[i][j]);
		}
		printf("\n");
	}
}

void print_matrix_rect(double **mat, int rows, int cols)
{
	for (int i = 0 ; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			printf("%lf\t", mat[i][j]);
		}
		printf("\n");
	}
}

void check_ans(double M[][DIM], double y[DIM], double b[DIM]) {
	int i, j;

	for(i = 0; i < DIM; i++) {
		double temp = 0.0;
		for(j = 0; j < DIM; j++) {
			temp += M[i][j] * y[j];
		}
		printf("%lf - %lf = %lf\n", temp, b[i], temp - b[i]);
	}

}


void check_matrixans_rect(double **arr1, double **arr2, double **arr3, int dim1, int dim2)
{
	printf("Checking correctness for rect matrix\n");
	int i, j, k;
	for(i = 0; i < dim1; i++) {
		for(j = 0; j < dim2; j++) {
			double temp = 0.0;
			for(k = 0; k < dim1; k++) {
				temp += arr1[i][k] * arr2[k][j];
			}
			printf("%lf - %lf = %lf\n", temp, arr3[i][j], temp - arr3[i][j]);
		}
	}
}


void check_matrixans(double M[][DIM], double y[][DIM], double b[][DIM]) {
	printf("Checking correctness for matrix\n");
	int i, j, k;
	for(i = 0; i < DIM; i++) {
		for(j = 0; j < DIM; j++) {
			double temp = 0.0;
			for(k = 0; k < DIM; k++) {
				temp += M[i][k] * y[k][j];
			}
			printf("%lf - %lf = %lf\n", temp, b[i][j], temp - b[i][j]);
		}
	}
}

void forward_sub_vector(double arr1[][DIM], double b[DIM], double y1[DIM]) {
	for(int i = 0; i < DIM; i++) {
		y1[i] = b[i];	
		for(int j = 0; j < i; j++) {
			y1[i] = y1[i] - arr1[i][j] * y1[j];
		}
		y1[i] = y1[i] / arr1[i][i];
	}
}

void backward_sub_vector(double arr2[][DIM], double b[DIM], double y2[DIM]) {
	for(int i = DIM - 1; i >= 0 ; i--) {
		y2[i] = b[i];
		for(int j = i + 1; j < DIM; j++) {
			y2[i] = y2[i] - arr2[i][j] * y2[j];
		}
		y2[i] = y2[i] / arr2[i][i];
	}
}

void matrix_forward_substitution_rectangular(double **, double **, double **, int, int);

int main(){
	srand(time(NULL));

	int i,j;
	int dim1 = 8, dim2 = 16;
	double **arr1, **arr2, **arr3;

	// first generate dim1xdim1 square matrix
	arr1 = new double*[dim1];
	for (int i = 0; i < dim1; i++)
		arr1[i] = new double[dim1];

	// now generate dim1xdim2 rectangular matrix
	arr2 = new double*[dim1];
	for (int i = 0; i < dim1; i++)
		arr2[i] = new double[dim2];

	// now generate dim1xdim2 rectangular matrix for answer
	arr3 = new double*[dim1];
	for (int i = 0; i < dim1; i++)
		arr3[i] = new double[dim2];

	int MOD = 10;
	int setter = 1;

	//initialize the first square matrix
	for (i = 0; i < dim1; i++)
	{
		for (j = 0; j < dim1; j++)
		{
			arr1[i][j] = 1.0 * setter;
			setter++;
		}
	}

	for (i = 0; i < dim1; i++)
		for (j = 0; j < dim1; j++)
			if (i < j)
				arr1[i][j] = 0.0;

	// initialize the second dim1, dim2 matrix
	for (i = 0; i < dim1; i++)
	{
		for (j = 0; j < dim2; j++)
		{
			arr2[i][j] = 1.0 * setter;
			setter++;
		}
	}

	printf("Matrix 1:\n");
	print_matrix_rect(arr1, dim1, dim1);
	printf("\n");

	printf("Matrix 2:\n");
	print_matrix_rect(arr2, dim1, dim2);

	/*
	 * Now forward and backward substitution based on matrices	
	 */	

	// First trying forward substitution for matrix
	/* for (int k = 0; k < dim2; k++) { // this is looping over columns of B matrix
		for (int i = 0; i < dim1; i++) {
			arr3[i][k] = arr2[i][k];	
			for (int j = 0; j < i; j++) {
				arr3[i][k] = arr3[i][k] - arr1[i][j] * arr3[j][k];
			}
			arr3[i][k] = arr3[i][k] / arr1[i][i];
		}
	} */
	matrix_forward_substitution_rectangular(arr1, arr2, arr3, dim1, dim2);

	printf("Matrix 3:\n");
	print_matrix_rect(arr3, dim1, dim2);

	check_matrixans_rect(arr1, arr3, arr2, dim1, dim2);

	// Now trying backward substitution for matrix
	/*for (int k = 0; k < dim2; k++) {
		for (int i = dim1 - 1; i >= 0 ; i--) {
			arr3[i][k] = B1[i][k];
			for(int j = i + 1; j < DIM; j++) {
				output2[i][k] = output2[i][k] - arr2[i][j] * output2[j][k];
			}
			output2[i][k] = output2[i][k] / arr2[i][i];
		}
	}
	check_matrixans(arr2, output2, B1);*/

	return 0;
}
