void vector_matrix_multiply(double *, double **, int , double *);

double vector_vector_multiply(double *, double *, int);

void get_cholesky(double **, double **, int);


void subtract_vec(double *, double *, double *, int);



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


