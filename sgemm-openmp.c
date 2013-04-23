#include <stdlib.h>
#include <nmmintrin.h>
#include <string.h>
#include <omp.h>

#define VERTICAL_ROLL 32
#define BLOCKSIZE 16

void sgemm( int m, int n, int d, float *A, float *C )
{
    int n1 = n+1, nEnd = n/VERTICAL_ROLL*VERTICAL_ROLL;
    float *B = A, *D = C;
	#pragma omp parallel for
	 for (int j = 0; j < n; j++) {
		int jn1 = j*(n+1), jn = j*n; float *Cjn = D+jn;
		// for (int b = 0; b < m; b+= BLOCKSIZE) {
			for (int i = 0; i < nEnd; i+=VERTICAL_ROLL) {
			    float *Cjni = Cjn+i;
			    float *Cjni1 = Cjni + 4;
			    float *Cjni2 = Cjni + 8;
			    float *Cjni3 = Cjni + 12;
			    float *Cjni4 = Cjni + 16;
			    float *Cjni5 = Cjni + 20;
			    float *Cjni6 = Cjni + 24;
			    float *Cjni7 = Cjni + 28;

			    int i1 = i+4;
			    int i2 = i+8;
			    int i3 = i+12;
			    int i4 = i+16;
			    int i5 = i+20;
			    int i6 = i+24;
			    int i7 = i+28;

			    __m128 Cij = _mm_loadu_ps(Cjni);
			    __m128 Cij1 = _mm_loadu_ps(Cjni1);
			    __m128 Cij2 = _mm_loadu_ps(Cjni2);
			    __m128 Cij3 = _mm_loadu_ps(Cjni3);
			    __m128 Cij4 = _mm_loadu_ps(Cjni4);
			    __m128 Cij5 = _mm_loadu_ps(Cjni5);
			    __m128 Cij6 = _mm_loadu_ps(Cjni6);
			    __m128 Cij7 = _mm_loadu_ps(Cjni7);


			    // for (int k = b; k < b+BLOCKSIZE && k < m; k++) {
			    for (int k = 0; k < m; k++) {
					int k1 = k + 1; float *Akn = B+k*n;
					__m128 Ajk = _mm_load1_ps(Akn+jn1);

					__m128 Aik = _mm_loadu_ps(Akn+i);
					__m128 Ai1k = _mm_loadu_ps(Akn+i1);
					__m128 Ai2k = _mm_loadu_ps(Akn+i2);
					__m128 Ai3k = _mm_loadu_ps(Akn+i3);
					__m128 Ai4k = _mm_loadu_ps(Akn+i4);
					__m128 Ai5k = _mm_loadu_ps(Akn+i5);
					__m128 Ai6k = _mm_loadu_ps(Akn+i6);
					__m128 Ai7k = _mm_loadu_ps(Akn+i7);

					Cij = _mm_add_ps(Cij, _mm_mul_ps(Ajk, Aik));
					Cij1 = _mm_add_ps(Cij1, _mm_mul_ps(Ajk, Ai1k));
					Cij2 = _mm_add_ps(Cij2, _mm_mul_ps(Ajk, Ai2k));
					Cij3 = _mm_add_ps(Cij3, _mm_mul_ps(Ajk, Ai3k));
					Cij4 = _mm_add_ps(Cij4, _mm_mul_ps(Ajk, Ai4k));
					Cij5 = _mm_add_ps(Cij5, _mm_mul_ps(Ajk, Ai5k));
					Cij6 = _mm_add_ps(Cij6, _mm_mul_ps(Ajk, Ai6k));
					Cij7 = _mm_add_ps(Cij7, _mm_mul_ps(Ajk, Ai7k));
			    }
			    _mm_storeu_ps(Cjni, Cij);
			    _mm_storeu_ps(Cjni1, Cij1);
			    _mm_storeu_ps(Cjni2, Cij2);
			    _mm_storeu_ps(Cjni3, Cij3);
			    _mm_storeu_ps(Cjni4, Cij4);
			    _mm_storeu_ps(Cjni5, Cij5);
			    _mm_storeu_ps(Cjni6, Cij6);
			    _mm_storeu_ps(Cjni7, Cij7);
			}
		// }
    }
    if (n % VERTICAL_ROLL != 0 && (n - (nEnd) >= 4)) {
		#pragma omp parallel for
		for (int j = 0; j < n; j++) {
			for (int i = nEnd; i < n/4*4; i+=4) {
				float *addrCij = D+i+j*n;
				float *Ajn1 = B+j*n1;
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
		    float *Ajn1 = B+j*n1;
		    for (int i = n/4*4; i < n; i++) {
			float *addrCij = D+i+j*n;
			float *Ajn1 = B+j*n1;
			float *Ai = B+i;
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
