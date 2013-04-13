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
	/* Matrix Padding */
	if (n % ROLL_SIZE != 0) {
		// TODO: padding code
	}
	for( int i = 0; i < n; i++ ) {
		a0 = _mm_loadu_ps(A+i+0*n);
		a1 = _mm_loadu_ps(A+i+n);
		a2 = _mm_loadu_ps(A+i+2*n);
		a3 = _mm_loadu_ps(A+i+3*n);
		a4 = _mm_loadu_ps(A+i+4*n);
		ab0 = _mm_loadu_ps(A+i+0*n+4);
		ab1 = _mm_loadu_ps(A+i+n+4);
		ab2 = _mm_loadu_ps(A+i+2*n+4);
		ab3 = _mm_loadu_ps(A+i+3*n+4);
		ab4 = _mm_loadu_ps(A+i+4*n+4);
    	for( int j = 0; j < n; j++ ) { 
    		__m128 cReg = _mm_loadu_ps(C+i+j*n);
      		for( int k = 0; k < m; k+= ROLL_SIZE ) { 
				cReg += A[i+k*(n)] * A[j*(n+1)+k*(n)];
				switch(k*ROLL_SIZE) {
					case 0:
						cReg+= _mm_mul_ps(a0, _mm_loadu_ps(A+j*(n+1)+k*(n)));
						cReg+= _mm_mul_ps(ab0, _mm_loadu_ps(A+j*(n+1)+k*(n)+4));
						break;
					case 1:
						cReg+= _mm_mul_ps(a1, _mm_loadu_ps(A+j*(n+1)+k*(n)));
						cReg+= _mm_mul_ps(ab1, _mm_loadu_ps(A+j*(n+1)+k*(n)+4));
						break;
					case 2:
						cReg+= _mm_mul_ps(a2, _mm_loadu_ps(A+j*(n+1)+k*(n)));
						cReg+= _mm_mul_ps(ab2, _mm_loadu_ps(A+j*(n+1)+k*(n)+4));
						break;
					case 3:
						cReg+= _mm_mul_ps(a3, _mm_loadu_ps(A+j*(n+1)+k*(n)));
						cReg+= _mm_mul_ps(ab3, _mm_loadu_ps(A+j*(n+1)+k*(n)+4));
						break;
					case 4:
						cReg+= _mm_mul_ps(a4, _mm_loadu_ps(A+j*(n+1)+k*(n)));
						cReg+= _mm_mul_ps(ab4, _mm_loadu_ps(A+j*(n+1)+k*(n)+4));
						break;
					default:
						cReg+= _mm_mul_ps(_mm_loadu_ps(A+i+k*(n)), _mm_loadu_ps(A+j*(n+1)+k*(n)));
						cReg+= _mm_mul_ps(_mm_loadu_ps(A+i+k*(n)+4), _mm_loadu_ps(A+j*(n+1)+k*(n)+4));
						break;
				}
			}
			_mm_storeu_ps(C+i+j*n, cReg);
		}
	}
}

