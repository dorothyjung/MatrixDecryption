#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <omp.h>

#define VERTICAL_ROLL 16
#define HORIZONTAL_ROLL 4
#define MIN(X, Y) (X<Y ? X : Y)

void sgemm( int m, int n, int d, float *A, float *C )
{
	int n1 = n+1;
	#pragma omp parallel for
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n/VERTICAL_ROLL*VERTICAL_ROLL; i+=VERTICAL_ROLL) {
			int i1 = i + 4;
			int i2 = i + 8;
			int i3 = i + 12;

			__m128 Cij = _mm_loadu_ps(C+i+j*n);
			__m128 Cij1 = _mm_loadu_ps(C+i1+j*n);
			__m128 Cij2 = _mm_loadu_ps(C+i2+j*n);
			__m128 Cij3 = _mm_loadu_ps(C+i3+j*n);

			for (int k = 0; k < m; k++) {
				 __m128 Ajk = _mm_load1_ps(A+j*n1+k*n);

				 __m128 Aik = _mm_loadu_ps(A+i+k*n);
				 __m128 Ai1k = _mm_loadu_ps(A+i1+k*n);
				 __m128 Ai2k = _mm_loadu_ps(A+i2+k*n);
				 __m128 Ai3k = _mm_loadu_ps(A+i3+k*n);

				 Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
				 Cij1 = _mm_add_ps(Cij1, _mm_mul_ps(Ajk, Ai1k));
				 Cij2 = _mm_add_ps(Cij2, _mm_mul_ps(Ajk, Ai2k));
				 Cij3 = _mm_add_ps(Cij3, _mm_mul_ps(Ajk, Ai3k));
			}
			_mm_store_ps(C+i+j*n, Cij);
			_mm_store_ps(C+i1+j*n, Cij1);
			_mm_store_ps(C+i2+j*n, Cij2);
			_mm_store_ps(C+i3+j*n, Cij3);
		}
	}
	if (n % VERTICAL_ROLL != 0 && n % 4 == 0) {
		for (int j = 0; j < n; j++) {
			for (int i = n/VERTICAL_ROLL*VERTICAL_ROLL; i < n; i+=4) {
				__m128 Cij = _mm_loadu_ps(C+i+j*n);
				for (int k = 0; k < m; k++) {
					 __m128 Ajk = _mm_load1_ps(A+j*n1+k*n);

					 __m128 Aik = _mm_loadu_ps(A+i+k*n);

					 Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
				}
				_mm_store_ps(C+i+j*n, Cij);
			}
		}
	}
	if (n % VERTICAL_ROLL != 0 && n % 4 != 0) {
		for (int j = 0; j < n; j++) {
			for (int i = n/VERTICAL_ROLL*VERTICAL_ROLL; i < n; i++) {
				__m128 Cij = _mm_load1_ps(C+i+j*n);
				for (int k = 0; k < m; k++) {
					__m128 Ajk = _mm_load1_ps(A+j*n1+k*n);
					__m128 Aik = _mm_load1_ps(A+i+k*n);
					Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
				}
				_mm_store_ss(C+i+j*n, Cij);
			}
		}
	}
}

