#!/bin/bash

JOBNAME=$PBS_JOBID

cd $PBS_O_WORKDIR
PATH=$PATH:$PBS_O_PATH

USERNAME=`whoami`
MASTER=`head -n1 $PBS_NODEFILE`
NODES=`sort -u $PBS_NODEFILE`

. ~/.bashrc
. ~/.bash_profile

echo $PBS_NODEFILE > nodes

echo "Username is " $USERNAME
echo "Master is " $MASTER
echo "List of nodes is " $NODES
echo "Also check out the nodes file"

echo "Spawning process on $HOSTNAME ($PBS_NODENUM of $PBS_NUM_NODES)"

HOST=`hostname`

echo "Hostname is " $HOST

module load gcc-4.9.2

if [ $MASTER.local = $HOSTNAME ] ; then
  echo $MASTER.local > masternode
fi

sleep 5

MASTER=`head -n1 masternode`
cat masternode

#if [ $HOST == "compute-0-29.local" ]; then
# /usr/lib64/openmpi/bin/mpirun -np 4 -x LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib:/usr/lib64/openmpi/lib:/usr/local/cuda/lib64:/home/15-418/opencv/build/lib:/home/15-418/glog-0.3.3/build/lib:/home/15-418/cudnn-6.5-linux-R1:/home/15-418/caffe/build/lib:/home/15-418/boost_1_57_0/stage/lib:/opt/opt-openmpi/1.8.5rc1/lib:/opt/gcc/4.9.2/lib64:/opt/gcc/4.9.2/lib:/opt/python/lib --hostfile hf -mca btl tcp,self -mca plm_rsh_agent ssh ./multinode_mpi_sample
#fi
# cuda-memcheck ./gp

#./cublas_matmul
#cuda-memcheck ./gp


time cuda-memcheck ./gp $HOSTNAME $MASTER $PBS_NUM_NODES > $HOSTNAME.log
