#include<vector>
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
                BCM(double**, double *, int,int, int);
                ~BCM();
                void set_BCM_log_hyperparam(double *);
                void get_BCM_log_hyperparam(double *); //FILL IN THE double *        

                void get_BCM_gradient_hyper(double *); //FILL in the double *
                double get_BCM_loglikelihood();
		void set_BCM_loghyper_eigen(Eigen::VectorXd initval);
		void get_loghyperparam(double *);
		void compute_BCM_test_means_and_var(double **, double *, double *, int );
		double get_BCM_negative_log_predprob(double *actual, double *predmean, double *predvar, int TS);
};

