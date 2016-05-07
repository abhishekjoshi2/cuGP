import sys
import numpy as np
ipfile = sys.argv[1]
numshard = int(sys.argv[2])
opprefix = sys.argv[3]
	
	
X = np.loadtxt(ipfile, delimiter = ",")

allsplits = np.array_split(X, numshard)

count = 0
for ind in allsplits:
	fl = opprefix + str(count) + ".txt"
	fptr = open(fl, "w")	
	for i in ind:	
		fptr.write(str(i) + "\n")
	fptr.close()
	count += 1

	
