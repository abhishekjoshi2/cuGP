gp: covkernel.o matrixops.o serial_gp.o
	g++ -g covkernel.o matrixops.o serial_gp.o -o gp -lm

covkernel.o: covkernel.cpp
	g++ -g -c covkernel.cpp -o covkernel.o

matrixops.o: matrixops.cpp
	g++ -g -c matrixops.cpp -o matrixops.o

serial_gp.o: serial_gp.cpp
	g++ -g -c serial_gp.cpp -o serial_gp.o

clean:
	rm -f *.o gp


