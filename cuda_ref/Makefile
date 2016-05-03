EXECUTABLE := ref

CU_FILES   := #cudaRenderer.cu

CC_FILES   := covkernel.cpp matrixops.cpp serial_gp.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
#CXX=g++ -m64
#CXX=mpic++
CXX=/usr/lib64/openmpi/bin/mpic++
MPIRUN=/usr/lib64/openmpi/bin/mpirun

CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)
MPI_LDFLAGS=-lpthread -lmpi -lmpi_cxx

LIBS       :=
FRAMEWORKS := 

ifeq ($(HOSTNAME), latedays.andrew.cmu.edu)
# Building on Latedays
NVCCFLAGS=-O3 -m64 -arch compute_20 -G -lineinfo
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib:/usr/lib64/openmpi/lib:$(shell printenv LD_LIBRARY_PATH)

else
# Building on Linux
NVCCFLAGS=-O3 -m64 -arch compute_20 -G -lineinfo
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/depot/cuda-7.5/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=main.o 


.PHONY: clean

default: $(EXECUTABLE)

clean:
	rm -rf $(OBJS) $(EXECUTABLE) $(LOGS) *.o

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

run_multinode_mpi_sample: multinode_mpi_sample
	$(MPIRUN) -np 4 -x LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) --hostfile hf -mca btl tcp,self -mca plm_rsh_agent ssh ./multinode_mpi_sample

multinode_mpi_sample: multinode_mpi_sample.cpp
	$(CXX) $(CXXFLAGS) $(MPI_LDFLAGS) multinode_mpi_sample.cpp -o multinode_mpi_sample $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)


main.o: main.cpp
	g++ -m64 main.cpp $(CXXFLAGS) -I/usr/local/depot/cuda-7.5/include/ -c -o main.o
	
main: main.o Utilities.o
	g++ -m64 main.o $(CXXFLAGS)   -I/usr/local/depot/cuda-7.5/include/ $(LDFLAGS) $(LDLIBS) -lcusolver -o main 
