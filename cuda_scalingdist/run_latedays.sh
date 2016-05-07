#!/bin/bash

mkdir -p old_logs

mv compute*.log old_logs

#qsub latedays.qsub -l nodes=1:ppn=24 -q titanx
qsub -l nodes=1:ppn=24 latedays.qsub
