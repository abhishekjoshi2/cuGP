#!/bin/bash

#qsub latedays.qsub -l nodes=1:ppn=24 -q titanx
qsub latedays.qsub -lwalltime=0:20:00 -l nodes=1:ppn=24 -q phi
