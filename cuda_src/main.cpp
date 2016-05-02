#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "../common/opcodes.h"

void run_kernel();

void run_gp();

void test_matrix_mult();

void setup();

void cg_solve();

void test_tmi();

int num_nodes, rank;

void *accept_commands(void *arg)
{
	int opcode;

	while (1)
	{
		printf("Node %d waiting for a command\n", rank);
		MPI::COMM_WORLD.Recv(&opcode, 1, MPI_INT, 0, 1);
		switch (opcode)
		{
			case COMPUTE_LOG_LIKELIHOOD:
				{
					printf("Node %d will start computing log likelihood\n", rank);
					//MPI::COMM_WORLD.Barrier();
					break;
				}

			case COMPUTE_GRADIENT_LOG_HYPERPARAMS:
				{
					printf("Node %d will start computing gradient log hyperparams\n", rank);
					//MPI::COMM_WORLD.Barrier();
					break;
				}

			case GET_LOGHYPERPARAMS:
				{
					printf("Node %d will start computing gradient log hyperparams\n", rank);
					//MPI::COMM_WORLD.Barrier();
					break;
				}

			case SET_LOGHYPERPARAMS:
				{
					printf("Node %d expecing log hyperparams to be set\n", rank);
					//MPI::COMM_WORLD.Barrier();
					break;
				}

			case DONE:
				{
					printf("Node %d got DONE message. Return!\n", rank);
					//MPI::COMM_WORLD.Barrier();
					return NULL;
				}
		}
	}
}

int main(int argc, char *argv[])
{
	/* char hostname[MPI_MAX_PROCESSOR_NAME];
	int len;
	pthread_t listener_thread;
	int done_opcode = DONE;

	MPI::Init(argc, argv);
	rank = MPI::COMM_WORLD.Get_rank();
	num_nodes = MPI::COMM_WORLD.Get_size();

	memset(hostname,0,MPI_MAX_PROCESSOR_NAME);
	MPI::Get_processor_name(hostname,len);
	memset(hostname+len,0,MPI_MAX_PROCESSOR_NAME-len);

	printf("Rank is %d and num_nodes is %d and hostname is %s\n", rank, num_nodes, hostname);

	int val = 5;
	for (int i = 0; i < 10; i++)
	{
		if (rank == 0)
		{
			for (int j = 1; j < num_nodes; j++)
			{
				printf("Node 0 sending %d to node %d\n", val, j);
				MPI::COMM_WORLD.Send(&val, 1, MPI_INT, j, 1);
			}
		}
		else
		{
			printf("Node %d trying to receive from node 0\n", rank);
			MPI::COMM_WORLD.Recv(&val, 1, MPI_INT, 0, 1);
		}
	}
	if (rank == 0)
	{
		printf("Node 0 polling for commands\n");
		pthread_create(&listener_thread, NULL, accept_commands, NULL);
		printf("Node %d running cg_solve\n", rank);

		cg_solve();

		printf("Solver done! Now tell everyone job is done.\n");
		for (int i = num_nodes - 1; i >= 0; i--)
		{
			MPI::COMM_WORLD.Send(&done_opcode, 1, MPI_INT, i, 1);
		}
	}
	else
	{
		printf("Node %d skipping cg_solve and polling for commands\n", rank);
		accept_commands(NULL);
	} 
	MPI::Finalize(); */

	setup(); //setting all the hyperparameters

	cg_solve();
	return 0;
}
