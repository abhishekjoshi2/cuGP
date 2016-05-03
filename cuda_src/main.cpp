#include <stdio.h>
#include <iostream>
#include <vector>
#include <cstdlib>
//#include <mpi.h>
#include "../common/opcodes.h"
#include "csapp.h"

void run_kernel();

void run_gp();

void test_matrix_mult();

void setup(int );

void cg_solve(char *);

void test_tmi();

int worker_id = 0;

void testing_phase(int offset, int numtest);

std::vector<int> worker_conn_fds;
int total_workers = 1;

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
					break;
				}

			case COMPUTE_GRADIENT_LOG_HYPERPARAMS:
				{
					printf("Node %s will start computing gradient log hyperparams\n", hostname);
					break;
				}

			case GET_LOGHYPERPARAMS:
				{
					printf("Node %s will start computing gradient log hyperparams\n", hostname);
					break;
				}

			case SET_LOGHYPERPARAMS:
				{
					printf("Node %s expecing log hyperparams to be set\n", hostname);
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
	int worker_id_counter = 0;

	listenfd = Open_listenfd (common_port);

	sleep(5);

	printf("Hostname %s is listening on port %s with listenfd = %d\n", argv[1], common_port, listenfd);

	if (strcmp(argv[1], "compute-0-37.local") == 0)
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
		connfd = Open_clientfd ("10.22.1.242", common_port);

		printf("Host %s connected to master, connfd is %d\n", argv[1], connfd);

		Rio_readn (connfd, (void *)&worker_id, sizeof(int));

		printf("Host %s got worker id as %d\n", argv[1], worker_id);
	}

	if (strcmp(argv[1], "compute-0-37.local") == 0)
	{
		printf("Master calling cg_solve()\n");
	
		int numtrain = 64;
		setup(numtrain);
		cg_solve(argv[1]);
		testing_phase(64, 64);
	}
	else
	{
		printf("Worker skipping cg_solve(), instead calling accept_commands\n");
		accept_commands(argv[1], connfd);
	}

	return 0;
}
