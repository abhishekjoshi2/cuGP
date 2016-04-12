import sys
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



def recurse(M):
	n = M.shape[0]
	if n == 1: # base case
		M[0,0] = np.sqrt(M[0,0])
		return
	M[0,0] = np.sqrt(M[0, 0])  # L_11
	M[1:n, 0] /= M[0,0]        # This is the vector L_21
	
	M[1:n, 1:n] -= np.outer(M[1:n, 0], M[1:n, 0])  # this is A_22 = A_22 - L_21 x L_21.transpose()
	recursive_chol(M[1:n, 1:n])

def recursive_chol(M):
	recurse(M)
	n = M.shape[0]
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
		print "Actual matrix"
		print symM
		# cholesky from numpy
		ans_ref = np.linalg.cholesky(symM)
		test_rec = np.copy(symM)
		recursive_chol(test_rec)
		serial_cholesky_inplace(symM)
		print '=========================='
		print "Output from serial_chol_inplace"
		print symM 
		#print np.dot(symM, symM.transpose())
		print "Ans from recursive chol"
		print test_rec
		print
		print
		print "------- test case", i+1, "--------"
		val = compare_two(ans_ref, symM)
		val = compare_two(ans_ref, test_rec)
		print val
		if val < 10e-6:
			print "Passed!"
		else:
			print "Failed"	

	
if __name__ == "__main__":
	main()


