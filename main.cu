#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <error.h>
#include <sys/time.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <string.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

int main(int argc, char **argv)
{
	int size = 1000;
	if (argc == 2) {
		size = atoi(argv[1]);
	}

	int *host_test = (int*)malloc(size * sizeof(int));
	int *dev_test;


	checkCudaErrors(cudaMalloc(&dev_test, size * sizeof(int)));
	thrust::device_ptr<int> thrust_test = thrust::device_pointer_cast(dev_test);
	thrust::fill(thrust_test, thrust_test + size, 1);
	thrust::device_vector<int> dev_res(size);

	struct timeval start, end;
	gettimeofday(&start, NULL);
	thrust::exclusive_scan(thrust_test, thrust_test + size, dev_res.begin());
	gettimeofday(&end, NULL);
	thrust::copy(dev_res.begin(), dev_res.end(), host_test);

	thrust_test[1] = 0;
//	for (int i = 0; i < size; i++) {
//		printf("%d ", host_test[i]);
//	}

	printf("\n");
	printf("Elapsed time for prefix sum: %ld usecs\n", (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec));

	return 0;
}
