#include <iostream>
#include <cmath>
#include "matrixops.h"

#define INPUT_FILE "input.txt"

double **generate_k(double **matrix, int n, int dim)
{
	return NULL;
}

int main()
{
	FILE *input_file;
	int n, dim;
	double **X;
	double **k_mat;

	input_file = fopen(INPUT_FILE, "r");

	if (input_file == NULL)
	{
		std::cout << "No input file found. Abort." << std::endl;
		return 0;
	}
	fscanf(input_file, "%d%d", &n, &dim);

	X = new double*[n];
	for (int i = 0; i < n; i++)
		X[i] = new double[dim];

	for (int i = 0; i < n; i++)
		for (int j = 0; j < dim; j++)
			fscanf(input_file, "%lf", &X[i][j]);

	k_mat = generate_k(X, n, dim);
}
