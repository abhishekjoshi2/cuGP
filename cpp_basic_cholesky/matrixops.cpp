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

// For computing Cholesky decomposition of a matrix A = LL'
//	- we first decompose input as L * L'
//	- output = L
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

// It computes 2 things:
//	- y'*inv(K)*y
//	- det(inv(k))
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

	matrix_transpose(L, U, n); 	//MAYBE WE CAN AVOID TRANSPOSE - BY USING L[j][i] INSTEAD OF U[i][j]

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

// For computing c = a - b (element wise vector subtraction)
void subtract_vec(double *a, double *b, double *c, int DIM){
	for (int i = 0 ; i < DIM ; i++){
		c[i] = a[i] - b[i];
	}
}

// For computing ans = a' * b;
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

// For computing: C = A - B (element wise matrix difference)
void subtract_matrices(double **A, double **B, double **C, int n1, int n2){
	for(int i = 0 ; i < n1; i++){
		for(int j = 0 ; j < n2 ; j++){
			C[i][j] = A[i][j] - B[i][j];		
		}
	}
}

// For computing: 
void get_outer_product(double *a, double *b, double **M, int n){
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ; j++){
			M[i][j] = a[i] * b[j];
		}
	}
}

// For computing: ans = inv(K) * y
// Much of it is taken from the above multiply_and_determinant code
void vector_Kinvy_using_cholesky(double **K, double *y, double *ans, int n){

	double **L, **U;

	L = new double*[n];
	U = new double*[n];
	for (int i = 0; i < n; i++)
	{
		L[i] = new double[n];
		U[i] = new double[n];
	}
	
	get_cholesky(K, L, n);
	matrix_transpose(L, U, n); 	//MAYBE WE CAN AVOID TRANSPOSE - BY USING L[j][i] INSTEAD OF U[i][j]

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
		ans[i] = temp[i];
		for (int j = i + 1; j < n; j++)
		{
			ans[i] -= U[i][j] * ans[j];
		}
		ans[i] /= U[i][i];
	}
	
	for(int i = 0 ; i < n ; i++){
		delete L[i];
		delete U[i];
	}
	delete L;
	delete U;
}

// For making M = I (identity matrix)
void make_identity(double **M, int n){
	for (int i = 0; i < n ; i++){
		for(int j = 0 ; j < n; j++){
			if(i == j) M[i][j] =1.0;
			else M[i][j] = 0.0;
		}
	}
}

// Computes output for satisfying A * output = B, using forward substitution (columnwise, for each column of B)
//	INVARIANT for correct result: A is lower triangular
void matrix_forward_substitution(double **A, double **B, double **output, int DIM){
	for(int k = 0; k < DIM ; k++){ // this is looping over columns of B matrix
                for(int i = 0 ; i < DIM; i++){
                        output[i][k] = B[i][k];
                        for(int j = 0 ; j < i ; j++){
                                output[i][k] = output[i][k] - A[i][j]*output[j][k];
                        }
                        output[i][k] = output[i][k] / A[i][i];
                }
        }
}

// Computes output for satisfying A * output = B, using backward substitution (columnwise, for each column of B)
//	INVARIANT for correct result: A is upper triangular
void matrix_backward_substitution(double **A, double **B, double **output, int DIM){
        for(int k = 0 ; k < DIM; k++){
                for(int i = DIM - 1; i >= 0 ; i--){
                        output[i][k] = B[i][k];
                        for(int j = i + 1; j < DIM; j++){
                                output[i][k] = output[i][k] - A[i][j]*output[j][k];
                        }
                        output[i][k] = output[i][k] / A[i][i];
                }
        }

}

//We need inverse(K), K = LL' => inv(K) = inv(LL') = inv(L') * inv(L)  equivalent to inv(L') * (inv(L) * I)
//	- Now, first we have to solve inv(L) * I
//		- we can use Matrix_forward_substitution (MFS)
//			=> let T = inv(L) * I <=> L * T = I (identity)
//			So we employ MFS with L and I to get T 
//
//		- our subsequent task is to solve: inv(L') * T
//			=> let S = inv(L') * T <=> L' * S = T
//			So we employ MBS with L' and T to get S					        
void compute_K_inverse(double **K, double **outputK, int n){

	double **temp1, **T, **I, **L;
	temp1 = new double*[n];
	T = new double*[n];
	I = new double *[n];
	L = new double *[n];
	
	for(int i = 0 ; i < n ;i++){
		temp1[i] = new double[n];
		T[i] = new double[n];
		I[i] = new double[n];
		L[i] = new double[n];
	}
	
	// 1. Solving inv(L) * I using MFS
	//	- Need identity matrix
	make_identity(I, n);
	// 	- Need the lower triangular matrix of K (by cholesky); result stored in L
	get_cholesky(K, L, n);
	//	- Now MFS
	matrix_forward_substitution(L, I, T, n); // should make L * T = I
	
		
	// 2. Solving inv(L') * T
	// 	- Need L.transpose()
	matrix_transpose(L, temp1, n); 		// temp1 = L'
	//	- Now MBS
	matrix_backward_substitution(temp1, T, outputK, n); // should make L' * outputK = T
	
}

void elementwise_matrixmultiply(double ** inp1, double ** inp2, double ** output, int n1, int n2){
	for(int i = 0 ; i < n1 ;i++){
		for(int j = 0 ; j < n2; j++){
			output[i][j] = inp1[i][j] * inp2[i][j];
		}
	}
}
