#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
	for(int j = 0; j < n; j = j++) {
		for(int k = 0; k < m; k++) {
			for(int i = 0; i < n; i = i + 4) {
				__m128 result = _mm_mul_ps(_mm_loadu_ps(A+i+k*n), _mm_loadu_ps(A+j*(n+1)+k*n));
				result = _mm_add_ps(_mm_loadu_ps(C+i+j*n), result);
				_mm_storeu_ps(C+i+j*n, result);
			}
		}
	}
}

