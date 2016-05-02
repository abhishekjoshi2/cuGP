EXECUTABLE := gp

EXECUTABLES := gp multinode_mpi_sample

CU_FILES   := #cudaRenderer.cu

CC_FILES   := covkernel.cpp matrixops.cpp serial_gp.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
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
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib:$(shell printenv LD_LIBRARY_PATH)

else
# Building on Linux
NVCCFLAGS=-O3 -m64 -arch compute_20 -G -lineinfo
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/depot/cuda-6.5/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=main.o cuda_gp.o cg_solver.o

.PHONY: clean

default: $(EXECUTABLE)

clean:
	rm -rf $(OBJS) $(EXECUTABLES) $(LOGS) *.o

$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

run_multinode_mpi_sample: multinode_mpi_sample
	$(MPIRUN) -np 4 -x LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) --hostfile hf -mca btl tcp,self -mca plm_rsh_agent ssh ./multinode_mpi_sample

multinode_mpi_sample: multinode_mpi_sample.cpp
	$(CXX) $(CXXFLAGS) $(MPI_LDFLAGS) multinode_mpi_sample.cpp -o multinode_mpi_sample $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

cuda_gp.o: cuda_gp.cu
	$(NVCC) cuda_gp.cu $(NVCCFLAGS) -c -o cuda_gp.o

main.o: main.cpp
	$(CXX) main.cpp $(CXXFLAGS) -c -o main.o

cg_solver.o: cg_solver.cpp
	$(CXX) cg_solver.cpp $(CXXFLAGS) -c -o cg_solver.o

cholesky_cu_solver: cholesky_cu_solver.cpp
	g++ -m64 $(CXXFLAGS) -o cholesky_cu_solver cholesky_cu_solver.cpp -I/usr/local/cuda/include $(LDFLAGS) $(LDLIBS) -lcusolver $(LDFRAMEWORKS)

tmi_cu_solver: tmi_cu_solver.cpp
	g++ -m64 $(CXXFLAGS) -o tmi_cu_solver tmi_cu_solver.cpp -I/usr/local/cuda/include $(LDFLAGS) $(LDLIBS) -lcusolver -lcublas $(LDFRAMEWORKS)
