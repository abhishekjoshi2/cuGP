EXECUTABLE := dgp

CU_FILES   := #cudaRenderer.cu

CC_FILES   := covkernel.cpp ../common/matrixops.cpp distributed_ver1.cpp BCM.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS := 


LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=distributed_ver1.o ../common/matrixops.o covkernel.o BCM.o


.PHONY: clean

default: $(EXECUTABLE)

clean:
		rm -rf $(OBJS) $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

BCM.o: BCM.cpp
	$(CXX) BCM.cpp $(CXXFLAGS) -c -o BCM.o

distributed_ver1.o: distributed_ver1.cpp
	$(CXX) distributed_ver1.cpp $(CXXFLAGS) -c -o distributed_ver1.o

matrixops.o: matrixops.cpp
	$(CXX) matrixops.cpp $(CXXFLAGS) -c -o matrixops.o

covkernel.o: covkernel.cpp
	$(CXX) covkernel.cpp $(CXXFLAGS) -c -o covkernel.o

#$%.o: %.cpp
#		$(CXX) $< $(CXXFLAGS) -c -o $@

#$%.o: %.cu
#		$(NVCC) $< $(NVCCFLAGS) -c -o $@
