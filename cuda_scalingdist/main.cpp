#include <stdio.h>
#include "../cuda_src/Eigen/Dense"

#include <iostream>
#include <vector>
#include <cstdlib>
//#include <mpi.h>
#include "../common/opcodes.h"
#include "csapp.h"
#include<string>
#include "../common/cycleTimer.h"

int numtrain = 0;
int numchunks = 0;
int dimensions = 0;

std::string prefix_input_file_name;
std::string prefix_label_file_name;

void destruct_cublas_cusoler();

void set_loghyper_eigen_multinode(Eigen::VectorXd initval);

void run_kernel();

void read_trainingdata_and_copy_to_GPU(std::string inputfilename, std::string labelfilename);

void run_gp();

void test_matrix_mult();

void setup(int, int);

void cg_solve(char *);

void test_tmi();

int worker_id = 0;

void testing_phase(int offset, int numtest);

double compute_log_likelihood();

void compute_gradient_log_hyperparams(double *localhp_grad);

double *get_loghyperparam();

void set_loghyper_eigen(Eigen::VectorXd initval);

std::vector<int> worker_conn_fds;

int total_workers = -1;

double *BCM_log_hyperparams;


void *accept_commands(char *hostname, int connfd)
{
	int opcode;

	while (1)
	{
		printf("Node %s waiting for a command\n", hostname);
		Rio_readn (connfd, (void *)&opcode, sizeof(int));

		switch (opcode)
		{
			case COMPUTE_LOG_LIKELIHOOD:
				{
					printf("Node %s will start computing log likelihood\n", hostname);
				
					double ll = 0.0;
			
					for(int i = worker_id; i < numchunks; i+=total_workers){ 
						std::string ipfile = prefix_input_file_name +  std::to_string(i) + std::string(".txt");
						std::string labfile = prefix_label_file_name +  std::to_string(i) + std::string(".txt");
						read_trainingdata_and_copy_to_GPU(ipfile, labfile);
						ll += compute_log_likelihood();
					}

					Rio_writen (connfd, (void *)&ll, sizeof(double));
					break;
				}

			case COMPUTE_GRADIENT_LOG_HYPERPARAMS:
				{
					printf("Node %s will start computing gradient log hyperparams\n", hostname);

					double grads[3] = {0.0};
					double temp[3];
				
					for(int i = worker_id; i < numchunks; i+=total_workers){ 
						std::string ipfile = prefix_input_file_name +  std::to_string(i) + std::string(".txt");
						std::string labfile = prefix_label_file_name +  std::to_string(i) + std::string(".txt");
						read_trainingdata_and_copy_to_GPU(ipfile, labfile);
						compute_gradient_log_hyperparams(temp);
						for(int j = 0 ; j < 3; j++){
							grads[j] += temp[j];
						}
					}	
					
					//compute_gradient_log_hyperparams(grads);

					Rio_writen (connfd, (void *)&grads[0], sizeof(double));
					Rio_writen (connfd, (void *)&grads[1], sizeof(double));
					Rio_writen (connfd, (void *)&grads[2], sizeof(double));
					break;
				}

			case GET_LOGHYPERPARAMS:
				{
					printf("Node %s will start returning log hyperparams\n", hostname);

					double *log_hyperparams = get_loghyperparam();

					// send the same value thrice
					Rio_writen (connfd, (void *)&log_hyperparams[0], sizeof(double));
					Rio_writen (connfd, (void *)&log_hyperparams[1], sizeof(double));
					Rio_writen (connfd, (void *)&log_hyperparams[2], sizeof(double));
					break;
				}

			case SET_LOGHYPERPARAMS:
				{
					printf("Node %s expecing log hyperparams to be set\n", hostname);
					double new_log_hyperparams[3];

					Rio_readn (connfd, (void *)&new_log_hyperparams[0], sizeof(double));
					Rio_readn (connfd, (void *)&new_log_hyperparams[1], sizeof(double));
					Rio_readn (connfd, (void *)&new_log_hyperparams[2], sizeof(double));

					Eigen::VectorXd new_eigen(3);

					new_eigen[0] = new_log_hyperparams[0];
					new_eigen[1] = new_log_hyperparams[1];
					new_eigen[2] = new_log_hyperparams[2];
					
					printf("------------------\n");
					printf("slave set the value of HP as\n");
					for(int i = 0; i < 3;i++){
						printf("%lf\n", new_eigen[i]);
					}
					printf("\n");
					printf("------------------\n");
					set_loghyper_eigen(new_eigen);

					break;
				}

			case DONE:
				{
					printf("Node %s got DONE message. Return!\n", hostname);
					return NULL;
				}
		}
	}
}

int main(int argc, char *argv[])
{
	int listenfd, connfd;
	char hostname[MAXLINE], port[MAXLINE];
	socklen_t clientlen;
	struct sockaddr_storage clientaddr;
	char *common_port = "15618";

	listenfd = Open_listenfd (common_port);

	sleep(5);

	total_workers = atoi(argv[3]);
	numchunks = atoi(argv[4]);
	numtrain = atoi(argv[5]);
	dimensions = atoi(argv[6]);
	prefix_input_file_name = std::string(argv[7]);
	prefix_label_file_name = std::string(argv[8]);
	

	printf("Hostname %s is listening on port %s with listenfd = %d\n", argv[1], common_port, listenfd);
	printf("Node is %s and Master is %s. Number of workers is %d\n", argv[1], argv[2], total_workers);
	printf("Number of shards (chunks) = %d\n", numchunks);
	printf("Number of traning points for each expert = %d, with D = %d\n", numtrain, dimensions);
	printf("Input file prefix: %s, Label file prefix = %s\n", prefix_input_file_name.c_str(), prefix_label_file_name.c_str());
		
	if (strcmp(argv[1], argv[2]) == 0)
	{
		for (int i = 0; i < total_workers - 1; i++)
		{
			clientlen = sizeof (clientaddr);

			// accept connections 
			connfd = Accept (listenfd, (SA *) & clientaddr, &clientlen);
			Getnameinfo ((SA *) & clientaddr, clientlen, hostname, MAXLINE,
					port, MAXLINE, 0);
			printf ("Accepted connection from (%s, %s). Connfd is %d\n", hostname, port, connfd);

			worker_conn_fds.push_back(connfd);

			int new_worker_id = i + 1;
			Rio_writen (connfd, (void *)&new_worker_id, sizeof(int));
		}
	}
	else
	{
		connfd = Open_clientfd (argv[2], common_port);

		printf("Host %s connected to master, connfd is %d\n", argv[1], connfd);

		Rio_readn (connfd, (void *)&worker_id, sizeof(int));

		printf("Host %s got worker id as %d\n", argv[1], worker_id);
	}


	if (strcmp(argv[1], argv[2]) == 0)
	{
		printf("Master calling cg_solve()\n");

		setup(numtrain, dimensions);

		BCM_log_hyperparams = new double[3];
		Eigen::VectorXd initval(3);                               
		for(int i = 0 ; i < 3; i++){                              
			initval[i] = 2.0;                                 
		}                                                         
		set_loghyper_eigen_multinode(initval);                    
		
		double startime = CycleTimer::currentSeconds();
                cg_solve(argv[1]);
                double endtime = CycleTimer::currentSeconds();
                printf("TOTAL training time = %lf\n", endtime - startime);
		destruct_cublas_cusoler();
		// testing_phase(numtrain,numtrain);
	}
	else
	{
		printf("Worker skipping cg_solve(), instead calling accept_commands\n");

		setup(numtrain, dimensions);

		accept_commands(argv[1], connfd);
	}

	return 0;
}
