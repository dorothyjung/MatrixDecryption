#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
	if (n % 4 != 0) {
		// pad matrix with 0s if not divisible by 4 
		float buffer[(n + n % 4) * (n + d)] = {0};
		for (int i = 0; i < d+n; i++) {
			memcpy(buffer + i*(n + n % 4), A + i*n, n);
		}
		A = buffer;
	}
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

