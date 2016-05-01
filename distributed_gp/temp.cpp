#include<iostream>
#include<cstdio>

int main(){

	int N = 100;
	int K = 1;
	
	int partition = N / K;
		
	int cursize = partition;
	int curstatval = 0;
	
	for(int i = 0 ; i < K ; i++){
		if(i == K - 1){
			cursize = N - curstatval;
		}
		int offset = curstatval;
		printf("offset = %d, cursize = %d\n", offset, cursize);
		curstatval += partition;	
	}
	return 0;
}
