import numpy as np


"""
Input: a square PD matrix
Output: Nothing (in place change with a lower triangular matrix (computed by Cholesky decom))
"""
def serial_cholesky_inplace(M):
	n = np.shape(M)[0]
	for col in range(n):
		M[col, col] = np.sqrt(M[col, col])
		for row in range(col + 1, n):
			M[row, col] = M[row, col] / M[col, col]
		
		for col2 in range(col+1, n):
			for row2 in range(col2, n):
				M[row2, col2] = M[row2, col2] - M[row2, col] * M[col2, col]
			
	for row in range(n):
		for col in range(row + 1, n):
			M[row, col] = 0.0

def serial_cholesky(M):
	n = np.shape(M)[0]
	newM = np.zeros((n ,n))

	for col in range(n):
		M[col, col] = np.sqrt(M[col, col])
		for row in range(col + 1, n):
			M[row, col] = M[row, col] / M[col, col]
		
		for col2 in range(col+1, n):
			for row2 in range(col2, n):
				M[row2, col2] = M[row2, col2] - M[row2, col] * M[col2, col]
			
	for row in range(n):
		for col in range(row + 1, n):
			M[row, col] = 0.0


def compare_two(M1, M2):
	return np.sum( (M1 - M2) ** 2) 

def main():
	
	#testing
	t = 10
	d = 4
	
	for i in range(t):
		M = np.random.random([d, d]) * 1000
		symM = np.dot(M.transpose(), M)
		print symM
		# cholesky from numpy
		ans_ref = np.linalg.cholesky(symM)
		serial_cholesky_inplace(symM)
		print '=========================='
		print symM 
		print np.dot(symM, symM.transpose())
		print
		print "------- test case", i+1, "--------"
		val = compare_two(ans_ref, symM)
		print val
		if val < 10e-6:
			print "Passed!"
		else:
			print "Failed"	

	
if __name__ == "__main__":
	main()


