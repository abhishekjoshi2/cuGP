#include <stdio.h>
#include "../cuda_src/Eigen/Dense"

#include <iostream>
#include <vector>
#include <cstdlib>
//#include <mpi.h>
#include "../common/cycleTimer.h"
#include "../common/opcodes.h"
#include "csapp.h"
#include <string>

void destruct_cublas_cusoler();
void set_loghyper_eigen_multinode(Eigen::VectorXd initval);

void run_kernel();

void run_gp();

void test_matrix_mult();

void setup(int, std::string, std::string );

void test_compute_K_train();

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
					
					double ll = compute_log_likelihood();

					Rio_writen (connfd, (void *)&ll, sizeof(double));
					break;
				}

			case COMPUTE_GRADIENT_LOG_HYPERPARAMS:
				{
					printf("Node %s will start computing gradient log hyperparams\n", hostname);

					double grads[3];

					compute_gradient_log_hyperparams(grads);

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

	printf("Hostname %s is listening on port %s with listenfd = %d\n", argv[1], common_port, listenfd);
	printf("Node is %s and Master is %s. Number of workers is %d\n", argv[1], argv[2], total_workers);
	
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

	int numtrain = 8192;
	int numtest = 1000;
	std::string prefix_input_file_name = "../chunked_dataset/siproper_9192_10_chunk";
	std::string prefix_label_file_name = "../chunked_dataset/siproper_9192_10_label";

	std::string ipfile = prefix_input_file_name + std::to_string(worker_id) + std::string(".txt");
	std::string opfile = prefix_label_file_name + std::to_string(worker_id) + std::string(".txt");

	if (strcmp(argv[1], argv[2]) == 0)
	{
		printf("Master calling cg_solve()\n");
		
		setup(numtrain, ipfile, opfile);
		
		BCM_log_hyperparams = new double[3];
		BCM_log_hyperparams[0] = 0.5;
		BCM_log_hyperparams[1] = 0.5;
		BCM_log_hyperparams[2] = 0.5;

		Eigen::VectorXd new_eigen(3);

		new_eigen[0] = BCM_log_hyperparams[0];
		new_eigen[1] = BCM_log_hyperparams[1];
		new_eigen[2] = BCM_log_hyperparams[2];

		set_loghyper_eigen_multinode(new_eigen);
		
		// test_compute_K_train();
		// return 0;
		double st = CycleTimer::currentSeconds();
		cg_solve(argv[1]);
		double end = CycleTimer::currentSeconds();
		printf("Total train time: %lf\n", end - st);

		/* testing_phase(numtrain,numtest);
		destruct_cublas_cusoler();	*/
	}
	else
	{
		printf("Worker skipping cg_solve(), instead calling accept_commands\n");

		setup(numtrain, ipfile, opfile);

		accept_commands(argv[1], connfd);
	}

	return 0;
}
