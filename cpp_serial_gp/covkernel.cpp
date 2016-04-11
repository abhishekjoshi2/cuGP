#include "covkernel.h"
#include "matrixops.h"
#include <cmath>
#include <iostream>
#include <cstdio>
#include "debug.h"
// Just for the sake of default constructor
Covsum::Covsum(){ }

// Initializing the values of the number of dimensions (d) and inputsize (n)
Covsum::Covsum(int n, int d){
	this->numdim = d;
	this->inputdatasize = n;
	this->loghyper = new double[3];
	
	this->tempKinv = new double* [n];
	this->tempKmatrix = new double* [n];
	this->tempWmatrix = new double* [n];
	this->tempAlphamatrix = new double* [n];
	this->tempmatrix2 = new double* [n];
	this->tempmatrix3 = new double* [n];
	this->tempmatrix4 = new double* [n];
	for (int i = 0 ; i < n ;i++){
		tempKinv[i] = new double[n];
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
		delete this->tempKinv[i];
		delete this->tempKmatrix[i];
		delete this->tempWmatrix[i];
		delete this->tempAlphamatrix[i];
		delete this->tempmatrix2[i];
		delete this->tempmatrix3[i];
		delete this->tempmatrix4[i];
	}
	delete this->tempKinv;
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

	if (debug)
	{
		std::cout << "ELL_SQ: " << ell_sq << std::endl;
		std::cout << "SIGNAL_VAR: " << signal_var << std::endl;
		std::cout << "NOISE_VAR: " << noise_var << std::endl;	
	}
	
	
	for (int i = 0 ; i < n ;i++)
	{
		for(int j = i; j < n;j++)
		{
			if (debug)
				printf("i = %d, j = %d \n", i, j);
			subtract_vec(X[i], X[j], this->temp1dvec, this->numdim);
			double val = dotproduct_vec(this->temp1dvec, this->temp1dvec, this->numdim);
			val = signal_var * exp(-val * 0.5 / ell_sq); 	//for SE kernel
			output[i][j] = val;
			output[j][i] = val;				// exploting symmetry

			if (i == j)
				output[i][j] += noise_var;		// for the noise covariance kernel
		}
	}
	if (debug)
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				std::cout << output[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}
	if (debug)
		printf("Sahi ho gaya\n");
}

// computes the 1 x n covariance vector for a given test point (xtest): CURRENTLY - only implements SE + noise
void Covsum::compute_k_test(double **X, double *xtest, double *output){
	double ell_sq = exp(this->loghyper[0] * 2); //l^2 after coverting back from the log form
	double signal_var = exp(this->loghyper[1] * 2); // signal variance
	double noise_var = exp(this->loghyper[2] * 2); //noise variance
	int n = this->inputdatasize;
	int d = this->numdim;
	for (int i = 0 ; i < n ; i++){
		subtract_vec(X[i], xtest, this->temp1dvec, d);
		double val = dotproduct_vec(this->temp1dvec, this->temp1dvec, this->numdim);
		output[i] = exp(-val * 0.5 * ell_sq);
	}
}

double Covsum::compute_loglikelihood(double **X, double *y){
	double ans = 0.0;
	int n = this->inputdatasize;

	compute_K_train(X, this->tempKmatrix);
	std::pair<double, double> pp = compute_chol_and_det(this->tempKmatrix, y, n);

	//if (debug)
		std::cout << "Product: " << pp.first << " log Determinant: " << pp.second << std::endl;

	return -0.5 * ( pp.first + pp.second + n * 1.83787);  // log (2 * pi) = 1.8378770664093453
	
}
void Covsum::compute_squared_dist(double **X, double c){
	int n = this->inputdatasize;
	int d = this->numdim;
	int i,j;
	
	if (debug)
		std::cout << "inside the squared dist function";

	for(i = 0 ; i < n ; i++){
		for(j = i ; j< n ;j++){
			if (debug)
				printf("i = %d, j = %d\n", i,j);
			if (i == j){
				tempmatrix2[i][j] = 0.0;
				continue;
			}
			subtract_vec(X[i], X[j], this->temp1dvec, d);
			double val = dotproduct_vec(this->temp1dvec, this->temp1dvec, d) / c; // remember c is l^2 for SE with noise
			tempmatrix2[i][j] = val;
			tempmatrix2[j][i] = val; 		// by symmetry
		}
	}
	if (debug)
		printf("Okay boyy: Now printing the squared distance matrix\n");
	print_matrix(tempmatrix2, n , n);	

}



//returns the array of gradients (size = num of hyperparameters)
double* Covsum::compute_gradient_loghyperparam(double **X, double *y){
	
	int n = this->inputdatasize;
	double ell_sq = exp(this->loghyper[0] * 2); //l^2 after coverting back from the log form
	double noise_var = exp(this->loghyper[2] * 2); //noise variance
	double *ans = new double[3]; // return 3 gradients

	if (debug)
		printf("Yahhan tak aya");
	compute_K_train(X, this->tempKmatrix); //required by all

	//for length scale	
	
	if (debug)
	{
		printf("compute_K_train ka calll ho gaya\n");
		printf("Now squred dist me\n");
	}
	compute_squared_dist(X, ell_sq);	 			   //tempmatrix2 will be populated

	if (debug)
		printf("\n\n Now calling elementwise matrix multiply\n");
	elementwise_matrixmultiply(this->tempKmatrix, this->tempmatrix2, this->tempmatrix3, n ,n ); //tempmatrix3 will be populated	
	
	//for signal variance
	//elementwise_constantmultiply(tempKmatrix, 2.0, tempmatrix4, n, n);  //not needed in current implementation, as will do this in later loop

	//for noise variance
	//tempmatrix5 = 2 * s2 * eye(n);  //not needed in current implementatio, as will do this in later loop 

	//--------------------------------
	// Now the common parts
		
	// computing inverse of K: tempKinv = K^{-1}

	if (debug)
		printf("Now calling compute_K_inverse\n");
	compute_K_inverse(this->tempKmatrix, this->tempKinv, n); // fill the value in tempKinv;	
	

	if (debug)
		printf("now Kinvy using cholesky to save some flops\n");
	vector_Kinvy_using_cholesky(this->tempKmatrix, y, this->temp1dvec, n); //fill the vector in temp1dvec = alpha
	
	// now computing: tempAlphamatrix =  alpha * alpha.transpose()
	
	if (debug)
	printf("now getting outer product of the 2 vectors\n");
	get_outer_product(this->temp1dvec, this->temp1dvec, this->tempAlphamatrix, n); //fill in the matrix tempAlphamatrix = t1 * t1.transpose()
	
	// now computing: tempWmatrix = K^{-1} - alpha * alpha.transpose() = tempKinv - tempAlphamatrix;
	
	if (debug)
		printf("now in subtract matrices\n");
	subtract_matrices(this->tempKinv, this->tempAlphamatrix, this->tempWmatrix, n , n);
	
	double psum1 = 0.0, psum2 = 0.0, psum3 = 0.0; //partial sum variables for ell, sigma_var, noise_var respectively


	if (debug)
		printf("Please check the W matrix\n");
	print_matrix(tempWmatrix, n, n);
	
	if (debug)
		printf("\n\n");

	if (debug)
		printf("now check K matrix\n");

	print_matrix(tempKmatrix, n, n);

	if (debug)
		printf("\n\n");
	
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ; j++){
			double curele = this->tempWmatrix[i][j];
			psum1 += curele * this->tempmatrix3[i][j]; //tempmatrix3 has the stuff of ell 
			psum2 += curele * 2.0 * this->tempKmatrix[i][j] ; // for sigma_var we need 2 * K
			if(i == j){
				psum3 += curele * noise_var * 2;
				psum2 -= curele * 2.0 * noise_var;
			}
		}
	}

	if (debug)
		printf(" KKKKKKKKKKKKKKKKKKKKKKKKK \n");
	ans[0] = psum1/2.0;
	ans[1] = psum2/2.0;
	ans[2] = psum3/2.0;
	return ans;
}


double* Covsum::get_loghyperparam(){
	return this->loghyper;
}

void Covsum::set_loghyperparam(double *initval){
	for(int i = 0 ; i < 3; i++) { //specific hardcoding for SE + NOISE -> REMVOVE 3
		this->loghyper[i] = initval[i];
	}
}

void Covsum::set_loghyper_eigen(Eigen::VectorXd initval){
	for(int i = 0 ; i < 3; i++) { //specific hardcoding for SE + NOISE -> REMVOVE 3
		this->loghyper[i] = initval[i];
	}
}

void Covsum::cg_solve(double **X_mat, double *y_vec, bool verbose=true) {
	
	const double INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket
	const double EXT = 3.0; // extrapolate maximum 3 times the current step-size
	const int MAX = 20;		// max 20 function evaluations per line search
	const double RATIO = 10;	// maximum allowed slope ratio
	const double SIG = 0.1, RHO = SIG/2;

	int n = 50;
	/* SIG and RHO are the constants controlling the Wolfe-
	   Powell conditions. SIG is the maximum allowed absolute ratio between
	   previous and new slopes (derivatives in the search direction), thus setting
	   SIG to low (positive) values forces higher precision in the line-searches.
	   RHO is the minimum allowed fraction of the expected (from the slope at the
	   initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
	   Tuning of SIG (depending on the nature of the function to be optimized) may
	  speed up the minimization; it is probably not worth playing much with RHO.
	*/

	/* The code falls naturally into 3 parts, after the initial line search is
	   started in the direction of steepest descent. 1) we first enter a while loop
	   which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
	   have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
	   enter the second loop which takes p2, p3 and p4 chooses the subinterval
	   containing a (local) minimum, and interpolates it, unil an acceptable point
	   is found (Wolfe-Powell conditions). Note, that points are always maintained
	   in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
	   conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
	   was a problem in the previous line-search. Return the best value so far, if
	   two consecutive line-searches fail, or whenever we run out of function
	   evaluations or line-searches. During extrapolation, the "f" function may fail
	   either with an error or returning Nan or Inf, and maxmize should handle this
	   gracefully.
	*/


	bool ls_failed = false;									//prev line-search failed
	// abj1 changing here
	// double f0 = -gp->log_likelihood();						//initial negative marginal log likelihood
	// Eigen::VectorXd df0 = -gp->log_likelihood_gradient();	//initial gradient
	// Eigen::VectorXd X = gp->covf().get_loghyper();			//hyper parameters

	double f0 = -1.0 * compute_loglikelihood(X_mat, y_vec);
	double *gradient_loghp = compute_gradient_loghyperparam(X_mat, y_vec);
	double *log_hyper_param = get_loghyperparam();

	Eigen::VectorXd df0(3);
	df0[0] = -1.0 * gradient_loghp[0];
	df0[1] = -1.0 * gradient_loghp[1];
	df0[2] = -1.0 * gradient_loghp[2];

	Eigen::VectorXd X(3);
	X[0] = log_hyper_param[0];
	X[1] = log_hyper_param[1];
	X[2] = log_hyper_param[2];

	if(verbose) std::cout << f0 << std::endl;

	Eigen::VectorXd s = -df0;								//initial search direction
	double d0 = -s.dot(s);									//initial slope
	double x3 = 1/(1-d0);

	double f3 = 0;
	double d3 = 0;
	Eigen::VectorXd df3 = df0;

	double x2 = 0, x4 = 0;
	double f2 = 0, f4 = 0;
	double d2 = 0, d4 = 0;

	for (unsigned int i = 0; i < n; ++i)
	{
		//copy current values
		Eigen::VectorXd X0 = X;
		double F0 = f0;
		Eigen::VectorXd dF0 = df0;
		unsigned int M = std::min(MAX, (int)(n-i));

		while(1)											//keep extrapolating until necessary
		{
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;
			df3 = df0;
			double success = false;

			while( !success && M>0)
			{
				M --;
				i++;

				/* gp->covf().set_loghyper(X+s*x3);
				f3 = -gp->log_likelihood();
				df3 = -gp->log_likelihood_gradient(); */

				set_loghyper_eigen((X+s*x3));
				f3 = -1.0 * compute_loglikelihood(X_mat, y_vec);
				double *df3_temp = compute_gradient_loghyperparam(X_mat, y_vec);

				df3[0] = -1.0 * df3_temp[0];
				df3[1] = -1.0 * df3_temp[1];
				df3[2] = -1.0 * df3_temp[2];

				if(verbose) std::cout << f3 << std::endl;

				bool nanFound = false;
				//test NaN and Inf's
				for (int j = 0; j < df3.rows(); ++j)
				{
					if(isnan(df3(j)))
					{
						nanFound = true;
						break;
					}
				}
				if(!isnan(f3) && !isinf(f3) && !nanFound)
					success = true;
				else
				{
					x3 = (x2+x3)/2; 						// if fail, bissect and try again
				}
			}
			//keep best values
			if(f3 < F0)
			{
				X0 = X+s*x3;
				F0 = f3;
				dF0 = df3;
			}

			d3 = df3.dot(s);								// new slope

			if( (d3 > SIG*d0) || (f3 >  f0+x3*RHO*d0) || M == 0) // are we done extrapolating?
			{
				break;
			}

			double x1 = x2; double f1 = f2; double d1 = d2;	// move point 2 to point 1
			x2 = x3; f2 = f3; d2 = d3;						// move point 3 to point 2
			double A = 6*(f1-f2) + 3*(d2+d1)*(x2-x1);				// make cubic extrapolation
			double B = 3*(f2-f1) - (2*d1+d2)*(x2-x1);
			x3 = x1-d1*(x2-x1)*(x2-x1)/(B+sqrt(B*B -A*d1*(x2-x1)));
			if(isnan(x3) || x3 < 0 || x3 > x2*EXT)			// num prob | wrong sign | beyond extrapolation limit
				x3 = EXT*x2;
			else if(x3 < x2+INT*(x2-x1))					// too close to previous point
				x3 = x2+INT*(x2-x1);
		}

		while( ( (std::abs(d3) > -SIG*d0) || (f3 > f0+x3*RHO*d0) ) && (M > 0))	// keep interpolating
		{
			if( (d3 > 0) || (f3 > f0+x3*RHO*d0) )			// choose subinterval
			{												// move point 3 to point 4
				x4 = x3;
				f4 = f3;
				d4 = d3;
			}
			else
			{
				x2 = x3;									//move point 3 to point 2
				f2 = f3;
				d2 = d3;
			}

			if(f4 > f0)
				x3 = x2 - (0.5*d2*(x4-x2)*(x4-x2))/(f4-f2-d2*(x4-x2));	// quadratic interpolation
			else
			{
				double A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);
				double B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
				x3 = x2+sqrt(B*B-A*d2*(x4-x2)*(x4-x2) -B)/A;
			}

			if(isnan(x3) || isinf(x3))
				x3 = (x2+x4)/2;

			x3 = std::max(std::min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2));

			/*gp->covf().set_loghyper(X+s*x3);
			f3 = -gp->log_likelihood();
			df3 = -gp->log_likelihood_gradient();*/

			set_loghyper_eigen((X+s*x3));
			f3 = -1.0 * compute_loglikelihood(X_mat, y_vec);
			double *df3_temp = compute_gradient_loghyperparam(X_mat, y_vec);

			df3[0] = -1.0 * df3_temp[0];
			df3[1] = -1.0 * df3_temp[1];
			df3[2] = -1.0 * df3_temp[2];

			if(f3 < F0)												// keep best values
			{
				X0 = X+s*x3;
				F0 = f3;
				dF0 = df3;
			}

			if(verbose) std::cout << F0 << std::endl;

			M--;
			i++;
			d3 = df3.dot(s);										// new slope
		}

		if( (std::abs(d3) < -SIG*d0) && (f3 < f0+x3*RHO*d0))
		{
			X = X+s*x3;
			f0 = f3;
			s = (df3.dot(df3)-df0.dot(df3)) / (df0.dot(df0))*s - df3;	// Polack-Ribiere CG direction
			df0 = df3;													// swap derivatives
			d3 = d0; d0 = df0.dot(s);
			if(verbose) std::cout << f0 << std::endl;
			if(d0 > 0)													// new slope must be negative
			{															// otherwise use steepest direction
				s = -df0;
				d0 = -s.dot(s);
			}

			x3 = x3 * std::min(RATIO, d3/(d0-std::numeric_limits< double >::min()));	// slope ratio but max RATIO
			ls_failed = false;																// this line search did not fail
		}
		else
		{														// restore best point so far
			X = X0;
			f0 = F0;
			df0 = dF0;

			if(verbose) std::cout << f0 << std::endl;

			if(ls_failed || i >= n)								// line search failed twice in a row
				break;											// or we ran out of time, so we give up

			s = -df0;
			d0 = -s.dot(s);										// try steepest
			x3 = 1/(1-d0);
			ls_failed = true;									// this line search failed
		}


	}
	set_loghyper_eigen(X);
}
