#include <cmath>
#include <utility>
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

void matrix_transpose(double **input, double**output, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			output[j][i] = input[i][j];
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

std::pair<double, double> multiply_and_get_determinant(double *yt, double **X, double *y, int n)
{
	double **L, **U;
	double product = 0.0;
	double det = 1;

	L = new double*[n];
	U = new double*[n];
	for (int i = 0; i < n; i++)
	{
		L[i] = new double[n];
		U[i] = new double[n];
	}
	
	get_cholesky(X, L, n);

	for (int i = 0; i < n; i++)
		det *= L[i][i] * L[i][i];

	matrix_transpose(L, U, n);

	// Ax = b -> LUx = b. Then y is defined to be Ux
	double *x = new double[n];
	double *temp = new double[n];
	// Forward solve Ly = b
	for (int i = 0; i < n; i++)
	{
		temp[i] = y[i];
		for (int j = 0; j < i; j++)
		{
			temp[i] -= L[i][j] * temp[j];
		}
		temp[i] /= L[i][i];
	}
	// Backward solve Ux = y
	for (int i = n - 1; i >= 0; i--)
	{
		x[i] = temp[i];
		for (int j = i + 1; j < n; j++)
		{
			x[i] -= U[i][j] * x[j];
		}
		x[i] /= U[i][i];
	}

	// now x has the product of X^-1 * y
	for (int i = 0; i < n; i++)
		product += yt[i] * x[i];

	std::pair<double, double> ret;

	ret = std::make_pair (product, det);

	delete x;
	delete temp;
	return ret;
}

/*

Calls to the matrix library:
- subtract_vec(a, b, c, DIM) -> c = a - b (all 3 are vectors of size DIM x 1)
                           for subtracting 2 vectors: takes 3 args - 2 inputs (both 1D double array)  and fills the third input array (1D double array)

- dotproduct_vec(a, b, DIM) ->  return a' * b
                        return: transpose(vector1) * vector2 [both vectors have size DIM x 1]
                        computes the dotproduct of 2 vectors : takes 2 args - 2 inputs (both 1D double array) and outputs a double

- compute_chol_and_det(K, y, n); -> returns a pair (note n is for size, as K: n x n and y: n x 1)
                                pair.first = transpose(y) * inverse(K) * inverse(y)
                                pair.second = determinant(K)
                                
- subtract_matrices(A, B, C, n1, n2); -> C = A - B  // all 3 matrices are of size n1 x n2
                                ( subtraction is elementwise )

- get_outer_product(a, b, M, n); ->  M = a * transpose(b)  //n is telling the size
                                basically it is vector1 * vector2.transpose()
                                a: n x 1, b = n x 1, transpose(b): 1 x n => M will be n x n

- compute_K_inverse(K, outputK, n); -> outputK = inverse(K) // so can't use cholesky, K is n x n square matrix

- vector_using_cholesky(K, y, ans, n); -> ans = inverse(K) * y //can very well use cholesky
                                        K: n x n, y: n x 1, ans: n x 1

*/

void subtract_vec(double *a, double *b, double *c, int DIM){
	for (int i = 0 ; i < DIM ; i++){
		c[i] = a[i] - b[i];
	}
}

double dotproduct_vec(double *a, double *b, int DIM){
	double ans  = 0.0;
	for(int i = 0 ; i < DIM; i++){
		ans += a[i] * b[i];
	}
	return ans;
}


std::pair<double, double> compute_chol_and_det(double **K, double * y, int n){
 	return multiply_and_get_determinant(y, K, y, n);
}

void subtract_matrices(double **A, double **B, double **C, int n1, int n2){
	for(int i = 0 ; i < n1; i++){
		for(int j = 0 ; j < n2 ; j++){
			C[i][j] = A[i][j] - B[i][j];		
		}
	}
}

void get_outer_product(double *a, double *b, double **M, int n){
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ; j++){
			M[i][j] = a[i] * b[j];
		}
	}
}


