#include <mpi.h>
#include <stdio.h>
#include <iostream>

class test {
	public:
	test() {
	  //printf("In constructor\n");
	}

	~test() {
	  //printf("In destructor\n");
	}
};

int main(int argc, char** argv) {
	int size;  /* the number of processes */
	int rank;  /* the id of the current process */
	int value;

	test obj;
	printf("Here\n");
	MPI::Init(argc, argv);

	rank = MPI::COMM_WORLD.Get_rank();// MPI_Comm_rank(MPI_COMM_WORLD, & rank);   // rank is the id of the current process
	size = MPI::COMM_WORLD.Get_size(); //(MPI_COMM_WORLD, & size);   // size is the total number of processes
	
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int len;

	memset(hostname,0,MPI_MAX_PROCESSOR_NAME);
	MPI::Get_processor_name(hostname,len);
	memset(hostname+len,0,MPI_MAX_PROCESSOR_NAME-len);

	printf("Rank is %d and size is %d and hostname is %s\n", rank, size, hostname);

	if (rank == 0) {
		for (int i = 1; i < size; i++) {
			value = i*i;
			printf("process 0 performing a blocking send of %d to process %d\n", value, i);
			MPI::COMM_WORLD.Send(& value, 1, MPI_INT, i, 1); 
		}
	} else {
		printf("Node %d waiting for receive\n", rank);
		MPI::COMM_WORLD.Recv(& value, 1, MPI_INT, MPI_ANY_SOURCE, 1);
		printf("Finished a blocking receive of %d from process 0\n", value);        
	}

	if (rank == 0) {
		for (int i = 1; i < size; i++) {
			value = i*i;
			printf("process 0 performing a blocking send of %d to process %d\n", value, i);
			MPI::COMM_WORLD.Send(& value, 1, MPI_INT, i, 1); 
		}
	} else {
		printf("Node %d waiting for receive\n", rank);
		MPI::COMM_WORLD.Recv(& value, 1, MPI_INT, MPI_ANY_SOURCE, 1);
		printf("Finished a blocking receive of %d from process 0\n", value);        
	}

	MPI::Finalize();
	return 0;
}
