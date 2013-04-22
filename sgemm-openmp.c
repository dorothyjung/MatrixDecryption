#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <omp.h>

#define VERTICAL_ROLL 16
#define HORIZONTAL_ROLL 4
#define MIN(X, Y) (X<Y ? X : Y)
#define BLOCKSIZE 16

void sgemm( int m, int n, int d, float *A, float *C )
{
    int n1 = n+1;
    int nEnd = n/VERTICAL_ROLL*VERTICAL_ROLL;
	#pragma omp parallel for
    for (int j = 0; j < n; j++) {
		int jn1 = j*(n+1), jn = j*n; float *Cjn = C+jn;
		for (int i = 0; i < nEnd; i+=VERTICAL_ROLL) {
		    int i1 = i + 4;
		    int i2 = i + 8;
		    int i3 = i + 12;
		    
		    __m128 Cij = _mm_loadu_ps(Cjn+i);
		    __m128 Cij1 = _mm_loadu_ps(Cjn+i1);
		    __m128 Cij2 = _mm_loadu_ps(Cjn+i2);
		    __m128 Cij3 = _mm_loadu_ps(Cjn+i3);
		    for (int k = 0; k < m; k++) {
				int k1 = k + 1; float *Akn = A+k*n;
				
				__m128 Ajk = _mm_load1_ps(Akn+jn1);
				
				__m128 Aik = _mm_loadu_ps(Akn+i);
				__m128 Ai1k = _mm_loadu_ps(Akn+i1);
				__m128 Ai2k = _mm_loadu_ps(Akn+i2);
				__m128 Ai3k = _mm_loadu_ps(Akn+i3);
				
				Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
				Cij1 = _mm_add_ps(Cij1, _mm_mul_ps(Ajk, Ai1k));
				Cij2 = _mm_add_ps(Cij2, _mm_mul_ps(Ajk, Ai2k));
				Cij3 = _mm_add_ps(Cij3, _mm_mul_ps(Ajk, Ai3k));
		    }
		    _mm_storeu_ps(Cjn+i, Cij);
		    _mm_storeu_ps(Cjn+i1, Cij1);
		    _mm_storeu_ps(Cjn+i2, Cij2);
		    _mm_storeu_ps(Cjn+i3, Cij3);
		}
    }
    if (n % VERTICAL_ROLL != 0 && (n - (n/VERTICAL_ROLL*VERTICAL_ROLL) >= 4)) {
		#pragma omp parallel for
		for (int j = 0; j < n; j++) {
			for (int i = nEnd; i < n/4*4; i+=4) {
				float *addrCij = C+i+j*n;
				float *Ajn1 = A+j*n1;
				float *Ai = A+i;				
				__m128 Cij = _mm_loadu_ps(addrCij);
				for (int k = 0; k < m; k++) {
				    int kn = k*n;				    
				    __m128 Ajk = _mm_load1_ps(Ajn1+k*n);
				    __m128 Aik = _mm_loadu_ps(Ai+k*n);
				    Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
				}
				_mm_storeu_ps(addrCij, Cij);
			}
		}
    }
    if ((n - nEnd) % 4 != 0) {
		#pragma omp parallel for
		for (int j = 0; j < n; j++) {
		    float *Ajn1 = A+j*n1;
		    for (int i = n/4*4; i < n; i++) {
				float *addrCij = C+i+j*n;
				float *Ajn1 = A+j*n1;
				float *Ai = A+i;
				__m128 Cij = _mm_loadu_ps(addrCij);
				for (int k = 0; k < m; k++) {
				    int kn = k*n;
				    __m128 Ajk = _mm_load1_ps(Ajn1+kn);
				    __m128 Aik = _mm_loadu_ps(Ai+kn);
				    Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
				}
				_mm_store_ss(addrCij, Cij);
		    }
		}	
	}	
}	

