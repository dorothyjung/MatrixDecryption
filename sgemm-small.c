#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>

#define ROLL_SIZE 8
#define BLOCK_SIZE 8
#define C(i) _mm_loadu_ps(C+i*ROLL_SIZE+j*n);
#define CV(i) _mm_loadu_ps(C+i*ROLL_SIZE+j*n+4);

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
	// float *temp;
	// if ((n % ROLL_SIZE) != 0) {
	// 	// pad matrix with 0s if not divisible by 4 
	//     float buffer[(n + ROLL_SIZE-(n % ROLL_SIZE)) * (n + d)];
	//     memset(buffer, 0, ((n + ROLL_SIZE-(n % ROLL_SIZE)) * (n + d))*sizeof(float));
	// 	for (int i = 0; i < d+n; i++) {
	// 		memcpy(buffer + i*n, A + i*n, n*(sizeof(float)));
	// 		//		    memcpy(buffer + i*(n + (n % ROLL_SIZE)), A + i*n, n*(sizeof(float)));
	// 	}
	// 	A = buffer;
	// 	temp = C;
	// 	float cbuffer[(n + ROLL_SIZE-(n % ROLL_SIZE)) * (n + ROLL_SIZE-(n % ROLL_SIZE))];
	// 	memset(cbuffer, 0, ((n + ROLL_SIZE-(n % ROLL_SIZE)) * (n + ROLL_SIZE-(n % ROLL_SIZE)))*sizeof(float));
	// 	C = cbuffer;
	// }
	for(int j = 0; j < n; j = j++) {
		__m128 c0 = C(0)
		__m128 c1 = C(1)
		__m128 c2 = C(2)
		__m128 c3 = C(3)
		__m128 c4 = C(4)
		__m128 cv0 = CV(0)
		__m128 cv1 = CV(1)
		__m128 cv2 = CV(2)
		__m128 cv3 = CV(3)
		__m128 cv4 = CV(4)
		for(int k = 0; k < m; k++) {
			__m128 transposeVector = _mm_load1_ps(A+j*(n+1)+k*n);
			for(int i = 0; i < n/ROLL_SIZE*ROLL_SIZE; i = i + ROLL_SIZE) {
				switch(i) {
					case 0*ROLL_SIZE: 
						c0 = _mm_add_ps(c0, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						cv0 = _mm_add_ps(cv0, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						break;
					case 1*ROLL_SIZE: 
						c1 = _mm_add_ps(c1, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						cv1 = _mm_add_ps(cv1, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						break;
					case 2*ROLL_SIZE: 
						c2 = _mm_add_ps(c2, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						cv2 = _mm_add_ps(cv2, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						break;
					case 3*ROLL_SIZE: 
						c3 = _mm_add_ps(c3, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						cv3 = _mm_add_ps(cv3, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						break;
					case 4*ROLL_SIZE: 
						c4 = _mm_add_ps(c4, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						cv4 = _mm_add_ps(cv4, _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector));
						break;
					default: 
						_mm_storeu_ps(C+i+j*n, _mm_add_ps(_mm_loadu_ps(C+i+j*n), _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector)));
						_mm_storeu_ps(C+i+j*n+4, _mm_add_ps(_mm_loadu_ps(C+i+j*n+4), _mm_mul_ps(_mm_loadu_ps(A+i+k*n+4), transposeVector)));
					}
			}
			if (n%ROLL_SIZE != 0) {
				for (int i = n/ROLL_SIZE*ROLL_SIZE; i < n; i++) {
					_mm_store_ss(C+i+j*n, _mm_add_ps(_mm_load_ss(C+i+j*n), _mm_mul_ps(_mm_load_ss(A+i+k*n), transposeVector)));
				}
			}
		}
		_mm_storeu_ps(C+0*ROLL_SIZE+j*n, c0);
		_mm_storeu_ps(C+1*ROLL_SIZE+j*n, c1);
		_mm_storeu_ps(C+2*ROLL_SIZE+j*n, c2);
		_mm_storeu_ps(C+3*ROLL_SIZE+j*n, c3);
		_mm_storeu_ps(C+4*ROLL_SIZE+j*n, c4);
		_mm_storeu_ps(C+0*ROLL_SIZE+j*n+4, cv0);
		_mm_storeu_ps(C+1*ROLL_SIZE+j*n+4, cv1);
		_mm_storeu_ps(C+2*ROLL_SIZE+j*n+4, cv2);
		_mm_storeu_ps(C+3*ROLL_SIZE+j*n+4, cv3);
		_mm_storeu_ps(C+4*ROLL_SIZE+j*n+4, cv4);
	}
	// //printMatrix(n, m, C);
	// if ( n % ROLL_SIZE != 0) {
	// 	for (int i = 0; i < n; i++) {
	// 		memcpy(temp + i*n, C + i*(n + ROLL_SIZE-(n % ROLL_SIZE)), n*(sizeof(float)));
	// 	}
	// }
}

