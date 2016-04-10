#include "covkernel.h"
#include "matrixops.h"
#include <cmath>
#include <iostream>

/*
class Covsum{
        private:

                double* loghyper;
                int inputdatasize; //number of training examples
                int numdim; // dimensionality of the problem
		double **tempKmatrix; //storing as a buffer to not allocate the n x n matrix again and again
 		double * temp1dvec; //storing as a buffer to not allocate the n x 1 vector again and again
		double ** tempmatrix2, \
		       ** tempmatrix3, \
		       ** tempmatrix4, \
			** tempWmatrix, \
			** tempAlphamatrix;
        public:

		Covsum();
                Covsum(int, int);
                ~Covsum();

                double compute_loglikelihood(double **, double *); // arguments are X, y
                double* compute_gradient_loghyperparam(double **, double *);
		void compute_K(double **, double **);
		void compute_k_test(double **, double *, double *);
		void Covsum::compute_squared_dist(double **, double );
             //   double* get_loghyperparam();
              //  void set_loghyperparam(double *);
}*/


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

- vector_Kinvy_using_cholesky(K, y, ans, n); -> ans = inverse(K) * y //can very well use cholesky
					K: n x n, y: n x 1, ans: n x 1

*/

// Just for the sake of default constructor
Covsum::Covsum(){ }

// Initializing the values of the number of dimensions (d) and inputsize (n)
Covsum::Covsum(int d, int n){
	this->numdim = d;
	this->inputdatasize = n;
	this->loghyper = new double[3];
		
	this->tempKmatrix = new double* [n];
	this->tempWmatrix = new double* [n];
	this->tempAlphamatrix = new double* [n];
	this->tempmatrix2 = new double* [n];
	this->tempmatrix3 = new double* [n];
	this->tempmatrix4 = new double* [n];
	for (int i = 0 ; i < n ;i++){
		tempKmatrix[i] = new double[n];
		tempWmatrix[i] = new double[n];
		tempAlphamatrix[i] = new double[n];
		tempmatrix2[i] = new double[n];
		tempmatrix3[i] = new double[n];
		tempmatrix4[i] = new double[n];
	}
	this->temp1dvec = new double[n];
}
Covsum::~Covsum(){
	delete this->loghyper;
	delete this->temp1dvec;
	for (int i = 0 ; i < this->inputdatasize ;i++){
		delete this->tempKmatrix[i];
		delete this->tempWmatrix[i];
		delete this->tempAlphamatrix[i];
		delete this->tempmatrix2[i];
		delete this->tempmatrix3[i];
		delete this->tempmatrix4[i];
	}
	delete this->tempKmatrix;
	delete this->tempWmatrix;
	delete this->tempAlphamatrix;
	delete this->tempmatrix2;
	delete this->tempmatrix3;
	delete this->tempmatrix4;
}

// Computes the n x n covariance matrix based on training inputs: CURRENTLY - only implements SE + NOISE
void Covsum::compute_K_train(double **X, double **output){
	double ell_sq = exp(this->loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(this->loghyper[1] * 2); // signal variance
	double noise_var = exp(this->loghyper[2] * 2); //noise variance
	int n = this->inputdatasize;

	for (int i = 0 ; i < n ;i++){
		for(j = i; i < n;j++){
			if (i == j){
				output[i][j] = noise_var;		// for the noise covariance kernel
				continue;
			}
			subtract_vec(X[i], X[j], temp1dvec, this->numdim);
			double val = dotproduct_vec(temp1dvec, temp1dvec, this->numdim);
			val = signal_var * exp(-val * 0.5 / ell_sq); 	//for SE kernel
			output[i][j] = val;
			output[j][i] = val;				// exploting symmetry
		}
	}
}

// computes the 1 x n covariance vector for a given test point (xtest): CURRENTLY - only implements SE + noise
void Covsum::compute_k_test(double **X, double *xtest, double *output){
	double ell_sq = exp(this->loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(this->loghyper[1] * 2); // signal variance
	double noise_var = exp(this->loghyper[2] * 2); //noise variance
	int n = this->inputdatasize;
	int d = this->numdim;
	for (int i = 0 ; i < n ; i++){
		subtract_vec(X[i], xtest, temp1dvec, d);
		double val = dotproduct_vec(temp1dvec, temp1dvec, this->numdim);
		output[i] = exp(-val * 0.5 * ell_sq);
	}
}

double Covsum::compute_loglikelihood(double **X, double *y){
	double ans = 0.0;
	int n = this->inputdatasize;

	compute_K_train(X, tempKmatrix);
	std::pair<double, double> pp = compute_chol_and_det(K, y, n);
	return -0.5 * ( pp.first + log(pp.second) + n * 1.83787);  // log (2 * pi) = 1.8378770664093453
	
}
void Covsum::compute_squared_dist(double **X, double c){
	int n = this->inputdimsize;
	int d = this->numdim;
	for(i = 0 ; i < n ; i++){
		for(j = i ; j< n ;j++){
			if (i == j){
				tempmatrix2[i][j] = 0.0;
				continue;
			}
			subtract_vec(X[i], X[j], temp1dvec, d);
			double val = dotproduct_vec(temp1dvec, temp1dvec, d) / c; // remember c is l^2 for SE with noise
			tempmatrix2[i][j] = val;
			tempmatrix2[j][i] = val; 		// by symmetry
		}
	}
}


//returns the array of gradients (size = num of hyperparameters)
double* Covsum::compute_gradient_loghyperparam(double **X, double *y){
	
	int n = this->inputdimsize;
	double ell_sq = exp(this->loghyper[0] * 2); //l^2 after coverting back from the log form
	double noise_var = exp(this->loghyper[2] * 2); //noise variance
	double *ans = new double[3]; // return 3 gradients

	compute_K_train(X, tempKmatrix); //required by all

	//for length scale	
	compute_squared_dist(X, ell_sq);	 			   //tempmatrix2 will be populated
	elementwise_matrixmultiply(tempKmatrix, tempmatrix2, tempmatrix3); //tempmatrix3 will be populated	
	
	//for signal variance
	//elementwise_constantmultiply(tempKmatrix, 2.0, tempmatrix4);  //not needed in current implementation, as will do this in later loop

	//for noise variance
	//tempmatrix5 = 2 * s2 * eye(n);  //not needed in current implementatio, as will do this in later loop 

		
	//--------------------------------
	// Now the common parts
		
	// computing inverse of K: tempKinv = K^{-1}
	compute_K_inverse(tempKmatrix, tempKinv, n); // fill the value in tempKinv;	
	vector_Kinvy_using_cholesky(tempKmatrix, y, temp1dvec, n); //fill the vector in temp1dvec = alpha
	
	// now computing: tempAlphamatrix =  alpha * alpha.transpose()
	get_outer_product(temp1dvec, temp1dvec, tempAlphamatrix, n); //fill in the matrix tempAlphamatrix = t1 * t1.transpose()
	
	// now computing: tempWmatrix = K^{-1} - alpha * alpha.transpose() = tempKinv - tempAlphamatrix;
	subtract_matrices(tempKinv, tempAlphamatrix, tempWmatrix, n , n);
	
	double psum1 = 0.0, psum2 = 0.0, psum3 = 0.0; //partial sum variables for ell, sigma_var, noise_var respectively
	
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ;i++){
			double curele = tempWmatrix[i][j];
			psum1 += curele * tempmatrix3[i][j]; //tempmatrix3 has the stuff of ell 
			psum2 += curele * 2.0 * tempKmatrix[i][j]; // for sigma_var we need 2 * K
			if(i == j){
				psum3 += curele * noise_var * 2;
			}
		}
	}

	ans[0] = psum1/2.0;
	ans[1] = psum2/2.0;
	ans[2] = psum3/2.0;
	return ans;
}

