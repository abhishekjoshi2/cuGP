#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>

#define DIM 4

void print_matrix(double m[][DIM] ,int rows, int cols){
        for (int i = 0 ; i < rows; i++){
                for(int j = 0; j < cols; j++){
                        printf("%lf\t", m[i][j]);
                }
                printf("\n");
        }
}

void check_ans(double M[][DIM], double y[DIM], double b[DIM]){
	int i, j;
	
	for(i = 0 ; i < DIM ; i++){
		double temp = 0.0;
		for(j = 0; j < DIM; j++){
			temp += M[i][j] * y[j];
		}
		printf("%lf - %lf = %lf\n", temp, b[i], temp - b[i]);
	}
	
}
int main(){

	srand(time(NULL));
	double arr1[DIM][DIM]; //lower triangular
	double arr2[DIM][DIM]; //upper triangular
	int i,j;

	int MOD = 10;

	//initialize
	for(i = 0 ; i < DIM; i++){
		for(j = 0 ; j < DIM; j++){
			if (i == j){
				arr1[i][j] = rand() % MOD + 1;
				arr2[i][j] = rand() % MOD + 1;
			}
			else if (i > j){ // for lower triangular matrix
				arr1[i][j] =  rand() % MOD + 1;
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
	for(i = 0 ; i < DIM ; i++){
		b[i] = rand() % MOD + 1;
		printf("%lf\n", b[i]);
	}
	
	double y1[DIM], y2[DIM];

	//forward substitution 
	// doing for arr1 and b to get y1: [arr1 * y1 = b]
	for(i = 0 ; i < DIM; i++){
		y1[i] = b[i];	
		for(j = 0 ; j < i ; j++){
			y1[i] = y1[i] - arr1[i][j]*y1[j];
		}
		y1[i] = y1[i] / arr1[i][i];
	}
	check_ans(arr1, y1, b);

	print_matrix(arr2, DIM, DIM);
	//backward substitution
	//doing for arr2 and b to get y2: [arr2 * y2 = b]
	for(i = DIM - 1; i >= 0 ; i--){
		y2[i] = b[i];
		for(j = i + 1; j < DIM; j++){
			y2[i] = y2[i] - arr2[i][j]*y2[j];
		}
		y2[i] = y2[i] / arr2[i][i];
		printf("%lf\n", y2[i]);
	}
	check_ans(arr2, y2, b);
	return 0;
}

