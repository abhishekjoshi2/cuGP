import sys
import numpy as np
ipfile = sys.argv[1]
numshard = int(sys.argv[2])
opprefix = sys.argv[3]
	
	
X = np.loadtxt(ipfile, delimiter = ",")

N, D = X.shape
ans = N/numshard
allsplits = np.array_split(X, numshard)

count = 0
for ind in allsplits:
	
	fptr = open(opprefix + str(count) + ".txt", "w")
	fptr.write(str(ans) + " " + str(D) + "\n")
	x,y = ind.shape
	
	for i in range(x):
		for j in range(y):
			fptr.write(str(ind[i][j]) + " ")
		fptr.write("\n")
	fptr.close()
	count += 1

	
