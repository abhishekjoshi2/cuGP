#include <iostream>
#include <cmath>

#define INPUT_FILE "input.txt"

void vector_matrix_multiply(double *vector, double **matrix, int n, double *out_vector)
{
	double sum;
	for (int k = 0; k < n; k++)
	{
		sum = 0.0;
		for (int i = 0; i < n; i++)
		{
			sum += vector[i] * matrix[i][k];
		}
		out_vector[k] = sum;
	}
}

double vector_vector_multiply(double *vector1, double *vector2, int n)
{
	double ret;
	for (int i = 0; i < n; i++)
		ret += vector1[i] * vector2[i];
	return ret;
}

void get_cholesky(double **input, double **output, int dim)
{
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			output[i][j] = input[i][j];

	for (int col = 0; col < dim; col++)
	{
		output[col][col] = std::sqrt(output[col][col]);

		for (int row = col + 1; row < dim; row++)
			output[row][col] = output[row][col] / output[col][col];

		for (int col2 = col + 1; col2 < dim; col2++)
			for (int row2 = col2; row2 < dim; row2++)
				output[row2][col2] = output[row2][col2] - output[row2][col] * output[col2][col];
	}

	for (int row = 0; row < dim; row++)
	{
		for (int col = row + 1; col < dim; col++)
			output[row][col] = 0.0;
	}

	/*for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			std::cout << output[i][j] << " " ;
		}
		std::cout << std::endl;
	}*/
}

double **generate_k(double **matrix, int n, int dim)
{
	return NULL;
}

void matrix_inverse_using_cholesky(double **input_mat, int n, double **output_mat)
{

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
