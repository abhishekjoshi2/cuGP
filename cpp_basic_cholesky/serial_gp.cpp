#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "matrixops.h"
#include "covkernel.h"

#define INPUT_FILE "input.txt"
#define LABEL_FILE "label.txt"
int main()
{
	FILE *input_file, *label_file;
	int n, dim;
	double **X;
	double **k_mat;
	double *y;
	
	input_file = fopen(INPUT_FILE, "r");
	label_file = fopen(LABEL_FILE, "r");

	if (input_file == NULL)
	{
		std::cout << "No input file found. Abort." << std::endl;
		return 0;
	}
	if (label_file == NULL)
	{
		std::cout << "No label file found. Abort." << std::endl;
		return 0;
	}

	// INput file has to have an additional line with n and d
	fscanf(input_file, "%d%d", &n, &dim);

	X = new double*[n];
	y = new double[n];

	for (int i = 0; i < n; i++)
		X[i] = new double[dim];

	for (int i = 0; i < n; i++)
		for (int j = 0; j < dim; j++)
			fscanf(input_file, "%lf", &X[i][j]);
	
	for(int i = 0 ; i < n ; i++){
		fscanf(label_file, "%lf", &y[i]);
	}

	double inithypervalues[] = {1.0, 1.0, 1.0};

	Covsum kernelobj(n, dim);
	kernelobj.set_loghyperparam(inithypervalues);
	double ans = kernelobj.compute_loglikelihood(X, y);
	
	std::cout << ans << std::endl;
	return 0;	
}		
	

