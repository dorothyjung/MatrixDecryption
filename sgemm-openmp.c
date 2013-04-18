#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <omp.h>

void sgemm(int m, int n, int d, float *A, float *C) {
    #pragma omp parallel
    {
	    for (k = 0; k < m; k++) {
	        for (j = 0; j < n; j++) {
	            for (i = 0; i < n; i++) {
	                
	            }
	        }
	    }
	}
}
