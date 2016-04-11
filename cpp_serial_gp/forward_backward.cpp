#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>

#define DIM 4

void print_matrix(double m[][DIM], int rows, int cols) {
	for (int i = 0 ; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			printf("%lf\t", m[i][j]);
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

int main(){

	srand(time(NULL));
	double arr1[DIM][DIM]; //lower triangular
	double arr2[DIM][DIM]; //upper triangular
	int i,j;

	int MOD = 10;

	//initialize
	for(i = 0; i < DIM; i++) {
		for(j = 0; j < DIM; j++) {
			if (i == j) {
				arr1[i][j] = rand() % MOD + 1;
				arr2[i][j] = rand() % MOD + 1;
			}
			else if (i > j) { // for lower triangular matrix
				arr1[i][j] = rand() % MOD + 1;
				arr2[i][j] = 0.0;
			}
			else { //for upper triangular matrix
				arr1[i][j] = 0.0;
				arr2[i][j] = rand() % MOD  + 1;
			}
		}
	}
	//print_matrix(arr1, DIM, DIM);
	//printf("\n");
	//print_matrix(arr2, DIM, DIM);

	double b[DIM];	
	for(i = 0; i < DIM; i++) {
		b[i] = rand() % MOD + 1;
	}

	double y1[DIM], y2[DIM];

	/*
	 * Basic forward and backward substitution based on vectors
	 */

	// forward substitution 
	// doing for arr1 and b to get y1: [arr1 * y1 = b]

	forward_sub_vector(arr1, b, y1);
	check_ans(arr1, y1, b);

	// backward substitution
	backward_sub_vector(arr2, b, y2);
	check_ans(arr2, y2, b);

	/*
	 * Now forward and backward substitution based on matrices	
	 */	

	double B1[DIM][DIM];
	double output1[DIM][DIM];
	double output2[DIM][DIM];

	for (i = 0; i < DIM; i++) {
		for (j = 0; j < DIM; j++) {
			B1[i][j] = rand() % MOD + 1;
		}
	}

	// First trying forward substitution for matrix
	for (int k = 0; k < DIM; k++) { // this is looping over columns of B matrix
		for (int i = 0; i < DIM; i++) {
			output1[i][k] = B1[i][k];	
			for(int j = 0; j < i; j++) {
				output1[i][k] = output1[i][k] - arr1[i][j] * output1[j][k];
			}
			output1[i][k] = output1[i][k] / arr1[i][i];
		}
	}

	check_matrixans(arr1, output1, B1);

	// Now trying backward substitution for matrix
	for (int k = 0; k < DIM; k++) {
		for (int i = DIM - 1; i >= 0 ; i--) {
			output2[i][k] = B1[i][k];
			for(int j = i + 1; j < DIM; j++) {
				output2[i][k] = output2[i][k] - arr2[i][j] * output2[j][k];
			}
			output2[i][k] = output2[i][k] / arr2[i][i];
		}
	}
	check_matrixans(arr2, output2, B1);

	return 0;
}
