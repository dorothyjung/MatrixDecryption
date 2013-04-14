#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>

#define ROLL_SIZE 8
#define BLOCK_SIZE 8

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
	/* float *temp; */
	/* int newsize = n + ROLL_SIZE - (n % ROLL_SIZE), width = n+d; */
	/* int i, j, k, isFringe = (n % ROLL_SIZE != 0); */
	/* if (isFringe) { */
	/* 	// pad matrix with 0s if not divisible by 4 */
	/*     float buffer[newsize * width]; */
	/*     memset(buffer, 0, (newsize * width)*sizeof(float)); */
	/*     for (i = 0; i < width; i++) { */
	/* 	    memcpy(buffer + i*newsize, A + i*n, n*(sizeof(float))); */
	/* 	} */
	/* 	A = buffer; */
	/* 	temp = C; */
	/* 	float cbuffer[newsize * newsize]; */
	/* 	memset(cbuffer, 0, (newsize * newsize)*sizeof(float)); */
	/* 	C = cbuffer; */
	/* } */
	/* for(j = 0; j < n; j++) { */
	/* 	for(k = 0; k < m; k++) { */
	/* 		__m128 transposeVector = _mm_load1_ps(A+j*(n+1)+k*n); */
	/* 		for(i = 0; i < n; i += ROLL_SIZE) { */
	/* 			_mm_storeu_ps(C+i+j*n, _mm_add_ps(_mm_loadu_ps(C+i+j*n), _mm_mul_ps(_mm_loadu_ps(A+i+k*n), transposeVector))); */
	/* 			_mm_storeu_ps(C+i+j*n+4, _mm_add_ps(_mm_loadu_ps(C+i+j*n+4), _mm_mul_ps(_mm_loadu_ps(A+i+k*n+4), transposeVector))); */
	/* 		} */
	/* 	} */
	/* } */
	/* //printMatrix(n, m, C); */
	/* if (isFringe) { */
	/* 	for (i = 0; i < n; i++) { */
	/* 	    memcpy(temp + i*n, C + i*newsize, n*(sizeof(float))); */

	/* Matrix Padding */
	if (n % ROLL_SIZE != 0) {
		// TODO: padding code
	}
	for( int i = 0; i < n; i++ ) {
		__m128 a0 = _mm_loadu_ps(A+i+0*n);
		__m128 a1 = _mm_loadu_ps(A+i+n);
		__m128 a2 = _mm_loadu_ps(A+i+2*n);
		__m128 a3 = _mm_loadu_ps(A+i+3*n);
		__m128 a4 = _mm_loadu_ps(A+i+4*n);
		__m128 ab0 = _mm_loadu_ps(A+i+0*n+4);
		__m128 ab1 = _mm_loadu_ps(A+i+n+4);
		__m128 ab2 = _mm_loadu_ps(A+i+2*n+4);
		__m128 ab3 = _mm_loadu_ps(A+i+3*n+4);
		__m128 ab4 = _mm_loadu_ps(A+i+4*n+4);
    	for( int j = 0; j < n; j++ ) { 
    		__m128 cReg = _mm_load1_ps(C+i+j*n);
      		for( int k = 0; k < m; k+= ROLL_SIZE ) { 
				switch(k*ROLL_SIZE) {
					case 0:
						cReg = _mm_add_ps(cReg, _mm_mul_ps(a0, _mm_loadu_ps(A+j*(n+1)+k*(n))));
						cReg = _mm_add_ps(cReg, _mm_mul_ps(ab0, _mm_loadu_ps(A+j*(n+1)+k*(n)+4)));
						break;
					case 1:
						cReg = _mm_add_ps(cReg, _mm_mul_ps(a1, _mm_loadu_ps(A+j*(n+1)+k*(n))));
						cReg = _mm_add_ps(cReg, _mm_mul_ps(ab1, _mm_loadu_ps(A+j*(n+1)+k*(n)+4)));
						break;
					case 2:
						cReg = _mm_add_ps(cReg, _mm_mul_ps(a2, _mm_loadu_ps(A+j*(n+1)+k*(n))));
						cReg = _mm_add_ps(cReg, _mm_mul_ps(ab2, _mm_loadu_ps(A+j*(n+1)+k*(n)+4)));
						break;
					case 3:
						cReg = _mm_add_ps(cReg, _mm_mul_ps(a3, _mm_loadu_ps(A+j*(n+1)+k*(n))));
						cReg = _mm_add_ps(cReg, _mm_mul_ps(ab3, _mm_loadu_ps(A+j*(n+1)+k*(n)+4)));
						break;
					case 4:
						cReg = _mm_add_ps(cReg, _mm_mul_ps(a4, _mm_loadu_ps(A+j*(n+1)+k*(n))));
						cReg = _mm_add_ps(cReg, _mm_mul_ps(ab4, _mm_loadu_ps(A+j*(n+1)+k*(n)+4)));
						break;
					default:
						cReg = _mm_add_ps(cReg, _mm_mul_ps(_mm_loadu_ps(A+i+k*(n)), _mm_loadu_ps(A+j*(n+1)+k*(n))));
						cReg = _mm_add_ps(cReg, _mm_mul_ps(_mm_loadu_ps(A+i+k*(n)+4), _mm_loadu_ps(A+j*(n+1)+k*(n)+4)));
						break;
				}
			}
			__m128 r1 = _mm_hadd_ps(cReg, cReg);
			__m128 r2 = _mm_hadd_ps(r1, r1);
			_mm_store_ss(C+i+j*n, r2);
		}
	}
}

