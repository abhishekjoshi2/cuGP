#!/bin/bash

mkdir -p old_logs

mv compute*.log old_logs

#qsub latedays.qsub -l nodes=1:ppn=24 -q titanx
qsub -lwalltime=0:20:00 -l nodes=2:ppn=24 latedays.qsub
