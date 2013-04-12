#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>

#define ROLL_SIZE 4

void printMatrix(int n, int m, float *A) {
    printf("Matrix: \n");
    for (int i = 0; i < m ; i++) {
	printf("=");
    }
    printf("\n");
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < m; j ++) {
	    printf("%.2f ", A[n*j+i]);
	}
	printf("\n");
    }
    for (int i = 0; i < m ; i++) {
	printf("=");
    }
    printf("\n");
}


void sgemm( int m, int n, int d, float *A, float *C )
{
	float *temp;
	if ((n % ROLL_SIZE) != 0) {
		// pad matrix with 0s if not divisible by 4 
	    float buffer[(n + (n % ROLL_SIZE)) * (n + d)];
	    memset(buffer, 0, ((n + (n % ROLL_SIZE)) * (n + d))*sizeof(float));
		for (int i = 0; i < d+n; i++) {
			memcpy(buffer + i*(n + (n % ROLL_SIZE)), A + i*n, n*(sizeof(float)));
		}
		A = buffer;
		temp = C;
		float cbuffer[(n + (n % ROLL_SIZE)) * (n + (n % ROLL_SIZE))];
		memset(cbuffer, 0, ((n + (n % ROLL_SIZE)) * (n + (n % ROLL_SIZE)))*sizeof(float));
		C = cbuffer;
	}
	for(int j = 0; j < n; j = j++) {
		for(int k = 0; k < m; k++) {
			for(int i = 0; i < n; i = i + ROLL_SIZE) {
				__m128 result = _mm_mul_ps(_mm_loadu_ps(A+i+k*n), _mm_load1_ps(A+j*(n+1)+k*n));
				result = _mm_add_ps(_mm_loadu_ps(C+i+j*n), result);
				_mm_storeu_ps(C+i+j*n, result);
			}
		}
	}
	printMatrix(n, m, C);
	if ( n % ROLL_SIZE != 0) {
		for (int i = 0; i < n; i++) {
			memcpy(temp + i*n, C + i*(n + n % ROLL_SIZE), n*(sizeof(float)));
		}
	}
}

