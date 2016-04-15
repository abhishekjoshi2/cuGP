
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
//#include <armadillo>
#define dim 6

void print_matrix(double m[][dim] ,int rows, int cols){
	for (int i = 0 ; i < rows; i++){
		for(int j = 0; j < cols; j++){
			printf("%lf\t", m[i][j]);
		}
		printf("\n");	
	}
}

int main()
{
	int t = 10, d = dim;
	double matrix1[dim][dim], matrix2[dim][dim];
	double matrix3[dim][dim] = { {0.0} };
	double matrix4[dim][dim] = { {0.0} };

/*
	{
	{ 945635.17700,   499512.018424,   912814.30729,   971589.33981},
 	{ 499512.018424,    415074.6312107,    698753.32253665,   669439.29627252},
 	{ 912814.30729558,   698753.32253665,  1316538.72005963,  1320391.02344577},
 	{ 971589.33981557,   669439.29627252,  1320391.02344577,  1388383.41214483}};

*/

/*
	{
	{ 1.55097672504, 1.54284928675, 0.844401118832, 0.884913338131},
	{ 1.54284928675, 1.88799888109, 1.08535041197, 1.11402077305 },
	{ 0.844401118832, 1.08535041197, 1.33038976826, 0.801101913691},
	{ 0.884913338131, 1.11402077305, 0.801101913691, 0.922611446652} };
*/


/*double matrix3[dim][dim] = 
{
{ 5329.000000, 4818.000000, 2190.000000, 4234.000000, 2628.000000, 2482.000000 },
{ 4818.000000, 4356.000000, 1980.000000, 3828.000000, 2376.000000, 2244.000000 },
{ 2190.000000, 1980.000000, 900.000000, 1740.000000, 1080.000000, 1020.000000 },
{ 4234.000000, 3828.000000, 1740.000000, 3364.000000, 2088.000000, 1972.000000 },
{ 2628.000000, 2376.000000, 1080.000000, 2088.000000, 1296.000000, 1224.000000 },
{ 2482.000000, 2244.000000, 1020.000000, 1972.000000, 1224.000000, 1156.000000 }
};*/
	int sum = 0;

	srand(time(NULL));
	for (int i = 0; i < dim; i++){
		for (int j = 0; j < dim; j++){
			matrix1[i][j] = rand() % 100 + 1;
			matrix2[j][i] = matrix1[i][j];
		}
	}

	//multiplyting matrix1 and matrix2 to generate matrix3 (which will be symmetric)
	for (int i = 0; i < dim; i++){
		for(int j = 0; j < dim; j++){
			for(int k = 0; k < dim; k++){
				matrix3[i][j] += matrix1[i][k]*matrix2[k][j];
			}
		}
	}	
		

	std::cout << "Original matrix is" << std::endl;	

	// if want to read from file, uncomment this
	/* FILE *fp = fopen("input.txt", "r");
	if (fp == NULL)
	{
		printf("Couldn't open file. Abort.\n");
		return 0;
	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			fscanf(fp, "%lf", &matrix3[i][j]); */

	print_matrix(matrix3, dim, dim);
	
	// cholesky for matrix3
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
	
	for (int i = 0; i < dim; i++){
		for(int j = 0; j < dim; j++){
			for(int k = 0; k < dim; k++){
				matrix4[i][j] += matrix3[i][k]*matrix3[j][k];
			}
		}
	}		

	
	std::cout << " Lower triangular matrix " << std::endl;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			std::cout << matrix3[i][j] << " " ;
		}
		std::cout << std::endl;
	}

	printf("\n");
	std::cout << " The reconstructed matrix is " << std::endl;
	
	print_matrix(matrix4, dim, dim);
	return 0;
}
