void run_kernel_cholesky(int dim)
{
	int start_id, b;
	int threads_per_block;
	int number_of_blocks;
	int num_iters;

	double startime, endtime ;
	start_id = 0;
//	dim = 5000;
	b = 2;
	start_id = 0;

	init_and_print();
//	get_input(dim);
	initialize_random(dim);
	printf("okay random bhar gaya\n");
	setup_cholesky(dim, b);
	
	// Input generation
	//	1. taking transpose of mt in mt_transpose
	threads_per_block = 1024;
	number_of_blocks = upit(dim * dim, threads_per_block);
	generic_matrix_transpose<<<number_of_blocks, threads_per_block>>>(mt, mt_transpose, dim, dim);
	cudaThreadSynchronize();
	printf("ab jakar transpose hua\n");
	/*
	print_matrix_kernel<<<1, 1>>>(mt, dim, dim);
	cudaThreadSynchronize();
	print_matrix_kernel<<<1, 1>>>(mt_transpose, dim, dim);
	cudaThreadSynchronize();
	*/
	
	startime = CycleTimer::currentSeconds();	
	dim3 blockDimTemp(32,32);
	dim3 gridDimTemp( upit(dim, blockDimTemp.x), upit(dim, blockDimTemp.y));
	//matrixmultiply_noshare(double *a, int rowsA, int colsA, double *b, int rowsB, int colsB, double *c)
	matrixmultiply_noshare<<<gridDimTemp, blockDimTemp >>>(mt, dim, dim,  mt_transpose, dim, dim, M);
	cudaThreadSynchronize();
	endtime = CycleTimer::currentSeconds();	
	printf("Now multiplication got over, total time taken for dim = %d, is %lf\n", dim, endtime - startime);

	// Now copying the symmetric matrix from CUDA to host
	orig_sym = new double[dim * dim];
	cudacall(cudaMemcpy(orig_sym, M,  sizeof(double) * dim * dim, cudaMemcpyDeviceToHost));
	
	printf("Host me aya kyaa??\n");
	
	// WRITING TO FILE
	/*
	std::ofstream out(filename);
	for(int i = 0; i < dim ; i++){
		for(int j = 0; j < dim ; j++){
			out << orig_sym[i*dim + j] << " ";
	//		printf("%lf ", orig_sym[i*dim + j]);
		}
		out << "\n";
	//	printf("\n");
	}
	out.close();
	*/	

	startime = CycleTimer::currentSeconds();
	num_iters = dim / b;
	for (int i = 0; i < num_iters; i++)
	{
		hardcoded_cholesky_2x2<<<1, 1>>>(M, a11, dim, b, start_id);
		cudaThreadSynchronize();

		if (i == num_iters - 1)
			break;

		// TODO optimize a21_transpose, by bypassing it perhaps? Can avoid transpose and manipulate indices inside next kernel
		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id) * b, threads_per_block);
		take_a21_transpose<<<number_of_blocks, threads_per_block>>>(M, a21_transpose, dim, b, start_id);
		cudaThreadSynchronize();

		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id), threads_per_block);
		forward_substitution_rectangular_a21<<<number_of_blocks, threads_per_block>>>(M, a11, a21_transpose, l21_transpose_from_fs, dim, b, start_id);
		cudaThreadSynchronize();

	//	printf("Printing l21_transpose_from_fs\n");
	//	print_matrix_kernel<<<1, 1>>>(l21_transpose_from_fs, b, dim - b - start_id);
	//	cudaThreadSynchronize();

		/*		
		printf("\n\n");
		printf(" ---------------------------------------- \n");	
		print_matrix_kernel<<<1, 1>>>(a11, b, b);
		cudaThreadSynchronize();
		printf(" ---------------------------------------- \n");
		print_matrix_kernel<<<1,1>>>(a21_transpose, b, dim - b - start_id);
		cudaThreadSynchronize();
		printf(" ---------------------------------------- \n");
		singlethread_temp_matmult_kernel<<<1, 1>>>(a11, a21_transpose, l21_transpose_from_fs, b, b, dim - b - start_id);
		cudaThreadSynchronize();
		print_matrix_kernel<<<1,1>>>(l21_transpose_from_fs, b, dim - b - start_id);
		cudaThreadSynchronize();
		printf("\n\n");
		*/
			
		//printf("\nNow printing entire M matrix\n");
		//print_matrix_kernel<<<1, 1>>>(M, dim, dim);
		//cudaThreadSynchronize();
		
		// TODO: Can include this tranpose in the forward_substitution_rectangular_a22 call!!!!
		// Now taking transpose of l21_transpose_from_fs
		 
		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id) * b, threads_per_block);
		generic_matrix_transpose<<<number_of_blocks, threads_per_block>>>(l21_transpose_from_fs, l21, b, dim - b - start_id);
		cudaThreadSynchronize();
		
//		printf("\nNow checking the transpose => \n");	
//		print_matrix_kernel<<<1,1>>>(l21, dim - b - start_id, b);
//		cudaThreadSynchronize();
//		printf("Checking the l21_transpose_from_fs matrix\n");
//		check_l21_kernel<<<1, 1>>>(a11, l21_transpose_from_fs, a21_transpose, b, b, dim - b - start_id);
//		cudaThreadSynchronize();

		//matrixmultiply_noshare<<<(double *a, int rowsA, int colsA, double *b, int rowsB, int colsB, double *c)
		int rowA = (dim - b - start_id) , colA = b, rowB = b , colB = (dim - b - start_id) ;
		dim3 blockDim(32,32);
		dim3 gridDim( upit(colB, blockDim.x), upit(rowA, blockDim.y));
		matrixmultiply_noshare<<<gridDim, blockDim >>>(l21, (dim - b - start_id), b,  l21_transpose_from_fs, b, dim - b - start_id, l22_temp);
		cudaThreadSynchronize();

		threads_per_block = 512;
		number_of_blocks = upit((dim - b - start_id) * (dim - b - start_id), threads_per_block);
		offseted_elementwise_subtraction<<<number_of_blocks, threads_per_block >>>(l22_temp, dim - b - start_id, M, dim, b, start_id);
		cudaThreadSynchronize();

		start_id += b;
	}
	// Fire a kernel for making upper-triangular as 0.0
	threads_per_block = 512;
	number_of_blocks = upit( (dim * dim), threads_per_block);
	set_upper_zero<<<number_of_blocks, threads_per_block>>>(M, dim);
	cudaThreadSynchronize();
	endtime = CycleTimer::currentSeconds();	
	printf("Totat time taken = %lf s\n", endtime - startime);	
	// Now checking!
	
	double *finalans = new double[dim * dim];
	cudacall(cudaMemcpy(finalans, M,  sizeof(double) * dim * dim, cudaMemcpyDeviceToHost));
	check_cholesky(finalans, orig_sym, dim);	

	/*for(int i = 0; i < dim ; i++){
		for(int j = 0; j < dim ; j++){
			printf("%lf ", finalans[i*dim + j]);
		}
		printf("\n");
	}*/
	
}
