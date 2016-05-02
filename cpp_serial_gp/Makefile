EXECUTABLE := gp

CU_FILES   := #cudaRenderer.cu

CC_FILES   := covkernel.cpp ../common/matrixops.cpp serial_gp.cpp

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

OBJS=serial_gp.o ../common/matrixops.o covkernel.o


.PHONY: clean

default: $(EXECUTABLE)

clean:
		rm -rf $(OBJS) $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

serial_gp.o: serial_gp.cpp
	$(CXX) serial_gp.cpp $(CXXFLAGS) -c -o serial_gp.o

matrixops.o: matrixops.cpp
	$(CXX) matrixops.cpp $(CXXFLAGS) -c -o matrixops.o

covkernel.o: covkernel.cpp
	$(CXX) covkernel.cpp $(CXXFLAGS) -c -o covkernel.o

#$%.o: %.cpp
#		$(CXX) $< $(CXXFLAGS) -c -o $@

#$%.o: %.cu
#		$(NVCC) $< $(NVCCFLAGS) -c -o $@
