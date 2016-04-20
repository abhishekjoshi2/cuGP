EXECUTABLE := gp

CU_FILES   := #cudaRenderer.cu

CC_FILES   := covkernel.cpp matrixops.cpp serial_gp.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS := 

ifeq ($(HOSTNAME), latedays.andrew.cmu.edu)
# Building on Latedays
NVCCFLAGS=-O3 -m64 -arch compute_20 -G -lineinfo
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
else
# Building on Linux
NVCCFLAGS=-O3 -m64 -arch compute_20 -G -lineinfo
LIBS += GL glut cudart
LDFLAGS=-L/usr/local/depot/cuda-6.5/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=main.o cuda_gp.o


.PHONY: clean

default: $(EXECUTABLE)

clean:
		rm -rf $(OBJS) $(EXECUTABLE) $(LOGS) *.o

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

cuda_gp.o: cuda_gp.cu
	$(NVCC) cuda_gp.cu $(NVCCFLAGS) -c -o cuda_gp.o

main.o: main.cpp
	$(CXX) main.cpp $(CXXFLAGS) -c -o main.o
