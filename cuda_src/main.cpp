#include <stdio.h>
#include<iostream>
#include<cstdlib>
#include "./Eigen/Dense"

void run_kernel();

void run_gp();

void test_matrix_mult();

void setup();

double compute_log_likelihood();
void compute_gradient_log_hyperparams(double *);
double *get_loghyperparam();
void set_loghyper_eigen(Eigen::VectorXd initval);
void cg_solve(){

	const double INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket
	const double EXT = 3.0; // extrapolate maximum 3 times the current step-size
	const int MAX = 20;		// max 20 function evaluations per line search
	const double RATIO = 10;	// maximum allowed slope ratio
	const double SIG = 0.1, RHO = SIG/2;

	int n = 100;
	bool ls_failed = false;									//prev line-search failed

	double f0 = -1.0 * compute_log_likelihood();
	double *gradient_loghp = new double [3];
	compute_gradient_log_hyperparams(gradient_loghp);

	double *log_hyper_param = get_loghyperparam();

	Eigen::VectorXd df0(3);
	df0[0] = 1.0 * gradient_loghp[0];
	df0[1] = 1.0 * gradient_loghp[1];
	df0[2] = 1.0 * gradient_loghp[2];

	Eigen::VectorXd X(3);
	X[0] = log_hyper_param[0];
	X[1] = log_hyper_param[1];
	X[2] = log_hyper_param[2];
	bool verbose = true;
	if (verbose) std::cout << f0 << std::endl;

	Eigen::VectorXd printkeliye = df0;								//initial search direction
	Eigen::VectorXd s = -df0;								//initial search direction
	double d0 = -s.dot(s);									//initial slope
	double x3 = 1/(1-d0);

	double f3 = 0;
	double d3 = 0;
	Eigen::VectorXd df3 =df0;

	double x2 = 0, x4 = 0;
	double f2 = 0, f4 = 0;
	double d2 = 0, d4 = 0;

	double *df3_temp = new double [3]; 
	for (int i = 0; i < n; ++i)
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

				printkeliye = X + s*x3;
				printf("\n\n PLEASE-SEE 1 : %lf, %lf, %lf\n\n", printkeliye[0], printkeliye[1], printkeliye[2]);
				set_loghyper_eigen((X+s*x3));
				f3 = -1.0 * compute_log_likelihood();
				compute_gradient_log_hyperparams(df3_temp);

				df3[0] = 1.0 * df3_temp[0];
				df3[1] = 1.0 * df3_temp[1];
				df3[2] = 1.0 * df3_temp[2];

				if(verbose) std::cout << f3 << std::endl;
				bool nanFound = false;
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

				printkeliye = X + s*x3;
				printf("\n\n PLEASE-SEE  2: %lf, %lf, %lf\n\n", printkeliye[0], printkeliye[1], printkeliye[2]);
			set_loghyper_eigen((X+s*x3));
			f3 = -1.0 * compute_log_likelihood();
			compute_gradient_log_hyperparams(df3_temp); 

			df3[0] = 1.0 * df3_temp[0];
			df3[1] = 1.0 * df3_temp[1];
			df3[2] = 1.0 * df3_temp[2];

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
			ls_failed = false;														// this line search did not fail
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
	printkeliye = X;
	printf("\n\n PLEASE-SEE  3: %lf, %lf, %lf\n\n", printkeliye[0], printkeliye[1], printkeliye[2]);
	set_loghyper_eigen(X);
}



int main()
{
	setup(); //setting all the hyperparameters

	run_gp();

	printf("\n\nnow running\n\n");
	
	cg_solve();
	//test_matrix_mult();
	// run_gp();
	return 0;
}
