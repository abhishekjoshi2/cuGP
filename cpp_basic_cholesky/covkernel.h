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
                        ** tempAlphamatrix, \
			**tempKinv;
        public:

                Covsum();
                Covsum(int, int);
                ~Covsum();

                double compute_loglikelihood(double **, double *); // arguments are X, y
                double* compute_gradient_loghyperparam(double **, double *);
                void compute_K_train(double **, double **);
                void compute_k_test(double **, double *, double *);
                void compute_squared_dist(double **, double );
                double* get_loghyperparam();
                void set_loghyperparam(double *);
};

