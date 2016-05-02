#include "covkernel.h"
#include "BCM.h"
#include<vector>
#include<cstdio>

/*
class BCM{
	private:
		double **full_input_dataset; // As name suggests: pointer to full inp dataset
		double *full_label_dataset;

		std::vector<Covsum *> Experts;

		int *offset; //for offset in each of the training set
	
		double *log_hyper_bcm; // log hyperparameters of the BCM
		double *gradient_hp_bcm; //gradient of log hyperparameters;
	
        	int num_experts; //stores the number of experts
	

	public:
                BCM(int, int);
                ~BCM();
                void set_BCM_log_hyperparam(double *);
                void get_BCM_log_hyperparam(double *); //FILL IN THE double *        

                void get_BCM_gradient_hyper(double *); //FILL in the double *
                double get_BCM_loglikelihood();
		void compute_BCM_test_means_and_var(double **, double *, double *, int );
		double get_BCM_negative_log_predprob(double *actual, double *predmean, double *predvar, int TS);
}; */

double BCM::get_BCM_negative_log_predprob(double *actual, double *predmean, double *predvar, int TS) {

        double ans = 0.0;
        for(int i = 0; i < TS; i++) {
                double val = 0.5 * log(6.283185 * predvar[i]) + pow( (predmean[i] - actual[i]) , 2) / (2 * predvar[i]);
                ans += val;
        }
        return ans / TS;
}


void product_of_experts(double **indmeans, double **indvar, double *tmeanvec, double *tvarvec, int size, double *loghyperparam, int numexp){
	
	for(int i  = 0 ; i < size ; i++){
	
		double tempvar = 0.0;
		double tempmean = 0.0;
		for(int E = 0 ; E < numexp; E++){
			double invvar = 1.0 / indvar[E][i];
			tempvar += invvar;
			tempmean += invvar * indmeans[E][i];
		}
		tempvar = 1.0 / tempvar;
		tempmean = tempvar * tempmean;
		
		tmeanvec[i] = tempmean;
		tvarvec[i] = tempvar;
	}
}

void BCM::compute_BCM_test_means_and_var(double **inputtestdata, double *tmeanvec, double *tvarvec, int size){
	// Now entities required for testing
	double **indiv_means;
	double **indiv_var;
	indiv_means = new double*[num_experts];
	indiv_var = new double*[num_experts];
	for(int i = 0 ; i < num_experts; i++){
		indiv_means[i] = new double [size];
		indiv_var[i] = new double [size];
	}	
	
	// now populating individual responses
	for(int i = 0 ; i < num_experts; i++){
		Experts[i]->compute_test_means_and_variances(full_input_dataset + offset[i], full_label_dataset + offset[i], inputtestdata, indiv_means[i], indiv_var[i], size);
	}

	//NOW: POE
	product_of_experts(indiv_means, indiv_var, tmeanvec, tvarvec, size, log_hyper_bcm, num_experts);	

}

BCM::BCM(double **inp, double *out, int N, int D, int K){

	full_input_dataset = inp;
	full_label_dataset = out;
	
	num_experts = K;

	int startvaloffset = 0;
	int partition = N / K;	
	int cursize = partition;
		
	offset = new int[K];
	log_hyper_bcm = new double[3];
	gradient_hp_bcm = new double[3];
	
	for (int i = 0; i < K;i++){
		if(i == K-1){
			cursize = N - startvaloffset;
		}
		offset[i] = startvaloffset;
		Covsum * temp = new Covsum(cursize, D);
		Experts.push_back(temp);
		startvaloffset += partition;
	}

}

BCM::~BCM(){
	/*
	delete[] offset;
	delete[] log_hyper_bcm;
	delete[] gradient_hp_bcm;
	
	for(int i = 0; i < num_experts; i++){
		delete Experts[i];
	}
	*/
}
void BCM::set_BCM_log_hyperparam(double *inputhp){
	for(int i = 0 ; i < 3 ; i++){
		log_hyper_bcm[i] = inputhp[i];
	}
	for(int i = 0; i < num_experts; i++){
		Experts[i]->set_loghyperparam(log_hyper_bcm);
	}	
}

void BCM::get_BCM_log_hyperparam(double *reqd){ //FILL IN THE double *        
	double *md = Experts[0]->get_loghyperparam();
	for(int i = 0 ; i < 3 ; i++){
		log_hyper_bcm[i] = md[i];
	}	
	for(int k = 1 ; k < num_experts; k++){
		md = Experts[k]->get_loghyperparam();
		for(int i = 0 ; i < 3 ; i++){
			log_hyper_bcm[i] += md[i];
		}	
	}
	//Now return
	for(int i = 0 ; i < 3 ; i++){
		reqd[i] = log_hyper_bcm[i];
	}	
}

void printit(double *a){
	printf("%lf %lf %lf\n", a[0], a[1], a[2]);
}

void BCM::get_BCM_gradient_hyper(double *reqd){ //FILL in the double *
	printf("\n\n----------------GRADIENT-COMPUTATION---------------\n");
	printf("The loghyper-params are: ");
	for(int i = 0 ; i < 3 ;i++){
		printf("%lf, ", log_hyper_bcm[i]);
	}
	printf("\n");
	
	double *md = Experts[0]->compute_gradient_loghyperparam(full_input_dataset + offset[0], full_label_dataset + offset[0]); 
	for(int i = 0 ; i < 3 ; i++){
		gradient_hp_bcm[i] = md[i];
	}
	printf("For expert 1: ");
	printit(md);	
	for(int k = 1 ; k < num_experts; k++){
		printf("For expert %d: ", k + 1);
		md = Experts[k]->compute_gradient_loghyperparam(full_input_dataset + offset[k], full_label_dataset + offset[k]);
		printit(md);
		for(int i = 0 ; i < 3 ; i++){
			gradient_hp_bcm[i] += md[i];
		}	
	}
	//Now return
	for(int i = 0 ; i < 3 ; i++){
		reqd[i] = gradient_hp_bcm[i];
	}		
	
}

double BCM::get_BCM_loglikelihood(){
	printf("-----------------LOG-LIKELIHOOD-COMPUTATION---------------\n");
	printf("The loghyper-params are: ");
	for(int i = 0 ; i < 3 ;i++){
		printf("%lf, ", log_hyper_bcm[i]);
	}
	printf("\n");	
	
		
	double ans = 0.0;
	for(int k = 0 ; k < num_experts; k++){
		double val =  Experts[k]->compute_loglikelihood(full_input_dataset + offset[k], full_label_dataset + offset[k]);
		ans = ans + val;
		printf("LL of Expert %d: %lf\n", k + 1, val);
	}
	return ans;
}

void BCM::get_loghyperparam(double *st){
	for(int i = 0 ; i < 3 ; i++){
		st[i] = log_hyper_bcm[i];
	}
}
void BCM::set_BCM_loghyper_eigen(Eigen::VectorXd initval) {
	for(int i = 0 ; i < 3 ; i++){
		log_hyper_bcm[i] = initval[i];
	}
	for(int i = 0; i < num_experts; i++){
		Experts[i]->set_loghyperparam(log_hyper_bcm);
	}	
}
