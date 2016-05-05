#include <stdio.h>
#include <vector>
#include<iostream>
#include<cstdlib>
#include "../cuda_src/Eigen/Dense"
//#include <mpi.h>

#include "csapp.h"
#include "../cuda_src/Eigen/src/Core/util/DisableStupidWarnings.h"
#include "../common/opcodes.h"

double compute_log_likelihood();
void compute_gradient_log_hyperparams(double *);
double *get_loghyperparam();
void set_loghyper_eigen(Eigen::VectorXd initval);

extern int total_workers;
extern std::vector<int> worker_conn_fds;

double compute_log_likelihood_multinode()
{
	int ll_opcode = COMPUTE_LOG_LIKELIHOOD;
	for (int i = 0; i < total_workers - 1; i++)
	{
		printf("\n\n");
		printf("Tell worker %d to compute log likelihood\n", i);

		Rio_writen (worker_conn_fds[i], (void *)&ll_opcode, sizeof(int));

	}
	// TODO now call getll on self
	double ll_sum = 0.0;

	for(int i = worker_id; i < numchunks; i+=total_workers){
		ipfile = prefix_input_file_name +  std::to_string(i) + std::string(".txt");
		labfile = prefix_label_file_name +  std::to_string(i) + std::string(".txt");
		read_trainingdata_and_copy_to_GPU(ipfile, labfile);
		ll_sum += compute_log_likelihood();
	}

	//ll_sum = compute_log_likelihood();

	for (int i = 0; i < total_workers - 1; i++)
	{
		double ll = 0.0;

		Rio_readn (worker_conn_fds[i], (void *)&ll, sizeof(double));
		printf("Got %lf as log likelihood from node %d\n", ll, i);
		ll_sum += ll;
	}
	printf("Final ll_sum is %lf\n", ll_sum);

	return ll_sum;
}

void compute_gradient_log_hyperparams_multinode(double *arg)
{
	int gradient_loghyper_opcode = COMPUTE_GRADIENT_LOG_HYPERPARAMS;
	for (int i = 0; i < total_workers - 1; i++)
	{
		printf("\n\n");
		printf("Tell worker %d to compute gradient of log hyperparams\n", i);

		Rio_writen (worker_conn_fds[i], (void *)&gradient_loghyper_opcode, sizeof(int));

	}

	double temp[3];
	//initializinig args;
	for(int j = 0; j < 3; j++) {
		arg[j] = 0.0;
	}
	for(int i = worker_id; i < numchunks; i+=total_workers){
		ipfile = prefix_input_file_name +  std::to_string(i) + std::string(".txt");
		labfile = prefix_label_file_name +  std::to_string(i) + std::string(".txt");
		read_trainingdata_and_copy_to_GPU(ipfile, labfile);
		compute_gradient_log_hyperparams(temp);
		for(int j = 0 ; j < 3; j++){
			arg[j] += temp[j];
		}
	}

	//compute_gradient_log_hyperparams(arg);

	for (int i = 0; i < total_workers - 1; i++)
	{
		double grad1, grad2, grad3;

		Rio_readn (worker_conn_fds[i], (void *)&grad1, sizeof(double));
		Rio_readn (worker_conn_fds[i], (void *)&grad2, sizeof(double));
		Rio_readn (worker_conn_fds[i], (void *)&grad3, sizeof(double));

		printf("Got %lf, %lf, %lf as gradient log hyperparams from node %d\n", grad1, grad2, grad3, i);
		arg[0] += grad1;
		arg[1] += grad2;
		arg[2] += grad3;
	}
	printf("Final gradients of log hyperparams are %lf, %lf, %lf\n", arg[0], arg[1], arg[2]);
}

double *get_loghyperparam_multinode()
{
	int get_loghyper_opcode = GET_LOGHYPERPARAMS;
	for (int i = 0; i < total_workers - 1; i++)
	{
		printf("\n\n");
		printf("Tell worker %d to get log hyperparams\n", i);

		Rio_writen (worker_conn_fds[i], (void *)&get_loghyper_opcode, sizeof(int));
	}

	//static double log_hyperparams[3] = {0.5, 0.5, 0.5};
	double *log_hyperparams = get_loghyperparam();
	for (int i = 0; i < total_workers - 1; i++)
	{
		double log_hyperparams_temp[3];

		Rio_readn (worker_conn_fds[i], (void *)&log_hyperparams_temp[0], sizeof(double));
		Rio_readn (worker_conn_fds[i], (void *)&log_hyperparams_temp[1], sizeof(double));
		Rio_readn (worker_conn_fds[i], (void *)&log_hyperparams_temp[2], sizeof(double));
		
		printf("Got %lf, %lf, %lf as log hyperparams from node %d\n", log_hyperparams_temp[0], log_hyperparams_temp[1], log_hyperparams_temp[2], i);

		log_hyperparams[0] += log_hyperparams_temp[0];
		log_hyperparams[1] += log_hyperparams_temp[1];
		log_hyperparams[2] += log_hyperparams_temp[2];
	}

	printf("Final log hyperparams are %lf, %lf, %lf\n", log_hyperparams[0], log_hyperparams[1], log_hyperparams[2]);
	return log_hyperparams;
}

void set_loghyper_eigen_multinode(Eigen::VectorXd initval)
{
	int set_loghyper_opcode = SET_LOGHYPERPARAMS;
	double new_loghyper_params[3];

	for (int i = 0; i < 3; i++)
		new_loghyper_params[i] = initval[i];

	for (int i = 0; i < total_workers - 1; i++)
	{
		printf("\n\n");
		printf("Tell node %d to set log hyperparams\n", i);

		Rio_writen (worker_conn_fds[i], (void *)&set_loghyper_opcode, sizeof(int));
	}

	set_loghyper_eigen(initval);
	printf("----------------------------");
	printf("the values are:- \n");
	for(int i = 0 ; i < 3 ;i++){

		printf("%lf\n", initval[i]);
	}
	printf("\n");
	printf("----------------------------");
	for (int i = 0; i < total_workers - 1; i++)
	{
		printf("Tell node %d to set log hyperparams\n", i);

		Rio_writen (worker_conn_fds[i], (void *)&new_loghyper_params[0], sizeof(double));
		Rio_writen (worker_conn_fds[i], (void *)&new_loghyper_params[1], sizeof(double));
		Rio_writen (worker_conn_fds[i], (void *)&new_loghyper_params[2], sizeof(double));
	}
}

void send_done_message()
{
	int done_opcode = DONE;
	for (int i = 0; i < total_workers - 1; i++)
	{
		printf("Tell node %d we are done!\n", i);

		Rio_writen (worker_conn_fds[i], (void *)&done_opcode, sizeof(int));
	}
}

void cg_solve(char *hostname)
{
	const double INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket
	const double EXT = 3.0; // extrapolate maximum 3 times the current step-size
	const int MAX = 20;		// max 20 function evaluations per line search
	const double RATIO = 10;	// maximum allowed slope ratio
	const double SIG = 0.1, RHO = SIG/2;

	int n = 100;
	bool ls_failed = false;									//prev line-search failed

	//double f0 = -1.0 * compute_log_likelihood();
double f0 = -1.0 * compute_log_likelihood_multinode();

	double *gradient_loghp = new double [3];

	//compute_gradient_log_hyperparams(gradient_loghp);
compute_gradient_log_hyperparams_multinode(gradient_loghp);

	//double *log_hyper_param = get_loghyperparam();
double *log_hyper_param = get_loghyperparam_multinode();

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

				//set_loghyper_eigen((X+s*x3));
set_loghyper_eigen_multinode((X+s*x3));

				//f3 = -1.0 * compute_log_likelihood();
f3 = -1.0 * compute_log_likelihood_multinode();

				//compute_gradient_log_hyperparams(df3_temp);
compute_gradient_log_hyperparams_multinode(df3_temp);

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

			//set_loghyper_eigen((X+s*x3));
set_loghyper_eigen_multinode((X+s*x3));

			//f3 = -1.0 * compute_log_likelihood();
f3 = -1.0 * compute_log_likelihood_multinode();

			// compute_gradient_log_hyperparams(df3_temp); 
compute_gradient_log_hyperparams_multinode(df3_temp);

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
	printf("\n\n%s: PLEASE-SEE  3: %lf, %lf, %lf\n\n", hostname, printkeliye[0], printkeliye[1], printkeliye[2]);
	// set_loghyper_eigen(X);
set_loghyper_eigen_multinode(X);
	
	send_done_message();
}
