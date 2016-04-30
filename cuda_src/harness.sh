#!/bin/bash

cd $PBS_O_WORKDIR
PATH=$PATH:$PBS_O_PATH
. ~/.bashrc
. ~/.bash_profile
HOST=`hostname`

echo "Hostname is " $HOST

module load gcc-4.9.2

#if [ $HOST == "compute-0-29.local" ]; then
# /usr/lib64/openmpi/bin/mpirun -np 4 -x LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib:/usr/lib64/openmpi/lib:/usr/local/cuda/lib64:/home/15-418/opencv/build/lib:/home/15-418/glog-0.3.3/build/lib:/home/15-418/cudnn-6.5-linux-R1:/home/15-418/caffe/build/lib:/home/15-418/boost_1_57_0/stage/lib:/opt/opt-openmpi/1.8.5rc1/lib:/opt/gcc/4.9.2/lib64:/opt/gcc/4.9.2/lib:/opt/python/lib --hostfile hf -mca btl tcp,self -mca plm_rsh_agent ssh ./multinode_mpi_sample
#fi
# cuda-memcheck ./gp
./gp
