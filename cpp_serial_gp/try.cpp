#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>

int main(){
	
	int n = 10;
	double **arr, \
	       **arr2;
	arr  = new double * [n];
	int i, j;
	for(i = 0; i < n ;i++){
		arr[i] = new double[n];
	}
	for(i = 0 ; i< n;i++){
		for (j = 0 ; j < n ;j++){
			arr[i][j] = rand() % 5;
//			printf("%lf ", arr[i][j]);
		}
//		printf("\n");
	}
	std::pair <double, double> pp;
	
	return 0;
}
