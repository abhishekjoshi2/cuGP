
/*"""
Input: a square PD matrix
Output: Nothing (in place change with a lower triangular matrix (computed by Cholesky decom))
"""
def serial_cholesky_inplace(M):

	n = np.shape(M)[0]
	for col in range(n):
		M[col, col] = np.sqrt(M[col, col])
		for row in range(col + 1, n):
			M[row, col] = M[row, col] / M[col, col]
		
		for col2 in range(col+1, n):
			for row2 in range(col2, n):
				M[row2, col2] = M[row2, col2] - M[row2, col] * M[col2, col]
			
	for row in range(n):
		for col in range(row + 1, n):
			M[row, col] = 0.0

def serial_cholesky(M):
	n = np.shape(M)[0]
	newM = np.zeros((n ,n))

	for col in range(n):
		M[col, col] = np.sqrt(M[col, col])
		for row in range(col + 1, n):
			M[row, col] = M[row, col] / M[col, col]
		
		for col2 in range(col+1, n):
			for row2 in range(col2, n):
				M[row2, col2] = M[row2, col2] - M[row2, col] * M[col2, col]
			
	for row in range(n):
		for col in range(row + 1, n):
			M[row, col] = 0.0


def compare_two(M1, M2):
	return np.sum( (M1 - M2) ** 2) 

def main():
	
	#testing
	t = 10
	d = 4
	
	for i in range(t):
		M = np.random.random([d, d])
		symM = np.dot(M.transpose(), M)
		
		# cholesky from numpy
		ans_ref = np.linalg.cholesky(symM)
		serial_cholesky_inplace(symM)
	        	
		print "------- test case", i+1, "--------"
		val = compare_two(ans_ref, symM)
		print val
		if val < 10e-6:
			print "Passed!"
		else:
			print "Failed"	
	*/

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>

#define dim 4
int main()
{
	int t = 10, d = dim;
	double matrix1[dim][dim], matrix2[dim][dim];
	double matrix3[dim][dim] =
	{
	{ 1.55097672504, 1.54284928675, 0.844401118832, 0.884913338131},
	{ 1.54284928675, 1.88799888109, 1.08535041197, 1.11402077305 },
	{ 0.844401118832, 1.08535041197, 1.33038976826, 0.801101913691},
	{ 0.884913338131, 1.11402077305, 0.801101913691, 0.922611446652} };
	int sum = 0;

	srand(time(NULL));
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			matrix1[i][j] = rand() % 5 + 1;

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			matrix2[j][i] = matrix1[i][j];

	for (int col = 0; col < dim; col++)
	{
		matrix3[col][col] = std::sqrt(matrix3[col][col]);

		for (int row = col + 1; row < dim; row++)
			matrix3[row][col] = matrix3[row][col] / matrix3[col][col];

		for (int col2 = col + 1; col2 < dim; col2++)
			for (int row2 = col2; row2 < dim; row2++)
				matrix3[row2][col2] = matrix3[row2][col2] - matrix3[row2][col] * matrix3[col2][col];

	}

	for (int row = 0; row < dim; row++)
	{
		for (int col = row + 1; col < dim; col++)
			matrix3[row][col] = 0.0;
	}

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			std::cout << matrix3[i][j] << " " ;
		}
		std::cout << std::endl;
	}

	return 0;
}
