#include<cstdlib>
#include<utility>
void vector_matrix_multiply(double *, double **, int , double *);

double vector_vector_multiply(double *, double *, int);

void get_cholesky(double **, double **, int);


void subtract_vec(double *, double *, double *, int);
double dotproduct_vec(double *, double *, int );
std::pair<double, double> compute_chol_and_det(double **, double *, int );
void subtract_matrices(double **, double **, double **, int , int );
void get_outer_product(double *, double *, double **, int );
void vector_Kinvy_using_cholesky(double **, double *, double *, int );
void make_identity(double **, int );
void matrix_forward_substitution(double **, double **, double **, int );
void matrix_backward_substitution(double **, double **, double **, int );
void compute_K_inverse(double **, double **, int );

void elementwise_matrixmultiply(double **, double **, double **, int , int);
