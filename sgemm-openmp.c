#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <omp.h>

#define VERTICAL_ROLL 8
#define HORIZONTAL_ROLL 4
#define VBLOCK 400
#define HBLOCK 32
#define TBLOCK 400
#define MIN(X, Y) (X<Y ? X : Y)

void sgemm( int m, int n, int d, float *A, float *C )
{
    int n1 = n+1;
	#pragma omp parallel for
	for (int t = 0; t < n; t+= TBLOCK) {
		for (int h = 0; h < m; h+= HBLOCK) {
			for (int v = 0; v < n/VERTICAL_ROLL*VERTICAL_ROLL; v+= VBLOCK ) {
				for (int j = t; j < MIN(t + TBLOCK, n); j++) {
					for (int k = h; k < MIN(h + HBLOCK, m); k+= HORIZONTAL_ROLL) {
						int k1 = k+1; 
						int k2 = k+2; 
						int k3 = k+3;
						// handle horizontal edge case
						__m128 Ajk, Ajk1, Ajk2, Ajk3, Ajk4;
						Ajk = _mm_load1_ps(A+j*n1+k*n);
						if (k1 < m) {
							Ajk1 = _mm_load1_ps(A+j*n1+k1*n);
							if (k2 < m) {
								Ajk2 = _mm_load1_ps(A+j*n1+k2*n);
								if (k3 < m) {
									Ajk3 = _mm_load1_ps(A+j*n1+k3*n);
								}else {
									Ajk3 = _mm_setzero_ps();
								}
							}else {
								Ajk2 = _mm_setzero_ps();
								Ajk3 = _mm_setzero_ps();
							}
						}else {
							Ajk1 = _mm_setzero_ps();
							Ajk2 = _mm_setzero_ps();
							Ajk3 = _mm_setzero_ps();
						}
						for (int i = v;  i < MIN(n/VERTICAL_ROLL*VERTICAL_ROLL, v + VBLOCK); i+=VERTICAL_ROLL) {
							int i1 = i+4;
							// int i2 = i+8;
							// int i3 = i+12;
							// int i4 = i+16;
							// int i5 = i+20;
							// int i6 = i+24;
							// int i7 = i+28;
							__m128 Cij = _mm_loadu_ps(C+i+j*n);
							__m128 Cij1 = _mm_loadu_ps(C+i1+j*n);
							// __m128 Cij2 = _mm_loadu_ps(C+i2+j*n);
							// __m128 Cij3 = _mm_loadu_ps(C+i3+j*n);
							// __m128 Cij4 = _mm_loadu_ps(C+i4+j*n);
							// __m128 Cij5 = _mm_loadu_ps(C+i5+j*n);
							// __m128 Cij6 = _mm_loadu_ps(C+i6+j*n);
							// __m128 Cij7 = _mm_loadu_ps(C+i7+j*n);

							__m128 Aik = _mm_loadu_ps(A+i+k*n);
							__m128 Ai1k = _mm_loadu_ps(A+i1+k*n);
							// __m128 Ai2k = _mm_loadu_ps(A+i2+k*n);
							// __m128 Ai3k = _mm_loadu_ps(A+i3+k*n);
							// __m128 Ai4k = _mm_loadu_ps(A+i4+k*n);
							// __m128 Ai5k = _mm_loadu_ps(A+i5+k*n);
							// __m128 Ai6k = _mm_loadu_ps(A+i6+k*n);
							// __m128 Ai7k = _mm_loadu_ps(A+i7+k*n);

							__m128 Aik1 = _mm_loadu_ps(A+i+k1*n);
							__m128 Ai1k1 = _mm_loadu_ps(A+i1+k1*n);
							// __m128 Ai2k1 = _mm_loadu_ps(A+i2+k1*n);
							// __m128 Ai3k1 = _mm_loadu_ps(A+i3+k1*n);
							// __m128 Ai4k1 = _mm_loadu_ps(A+i4+k1*n);
							// __m128 Ai5k1 = _mm_loadu_ps(A+i5+k1*n);
							// __m128 Ai6k1 = _mm_loadu_ps(A+i6+k1*n);
							// __m128 Ai7k1 = _mm_loadu_ps(A+i7+k1*n);

							__m128 Aik2 = _mm_loadu_ps(A+i+k2*n);
							__m128 Ai1k2 = _mm_loadu_ps(A+i1+k2*n);
							// __m128 Ai2k2 = _mm_loadu_ps(A+i2+k2*n);
							// __m128 Ai3k2 = _mm_loadu_ps(A+i3+k2*n);
							// __m128 Ai4k2 = _mm_loadu_ps(A+i4+k2*n);
							// __m128 Ai5k2 = _mm_loadu_ps(A+i5+k2*n);
							// __m128 Ai6k2 = _mm_loadu_ps(A+i6+k2*n);
							// __m128 Ai7k2 = _mm_loadu_ps(A+i7+k2*n);

							__m128 Aik3 = _mm_loadu_ps(A+i+k3*n);
							__m128 Ai1k3 = _mm_loadu_ps(A+i1+k3*n);
							// __m128 Ai2k3 = _mm_loadu_ps(A+i2+k3*n);
							// __m128 Ai3k3 = _mm_loadu_ps(A+i3+k3*n);
							// __m128 Ai4k3 = _mm_loadu_ps(A+i4+k3*n);
							// __m128 Ai5k3 = _mm_loadu_ps(A+i5+k3*n);
							// __m128 Ai6k3 = _mm_loadu_ps(A+i6+k3*n);
							// __m128 Ai7k3 = _mm_loadu_ps(A+i7+k3*n);

							_mm_storeu_ps(C+i+j*n, _mm_add_ps(Cij, _mm_add_ps(_mm_mul_ps(Aik3, Ajk3), _mm_add_ps(_mm_mul_ps(Aik2, Ajk2), _mm_add_ps(_mm_mul_ps(Aik1, Ajk1), _mm_mul_ps(Aik, Ajk))))));
							_mm_storeu_ps(C+i1+j*n, _mm_add_ps(Cij1, _mm_add_ps(_mm_mul_ps(Ai1k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai1k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai1k, Ajk), _mm_mul_ps(Ai1k1, Ajk1))))));
							// _mm_storeu_ps(C+i2+j*n, _mm_add_ps(Cij2, _mm_add_ps(_mm_mul_ps(Ai2k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai2k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai2k, Ajk), _mm_mul_ps(Ai2k1, Ajk1))))));
							// _mm_storeu_ps(C+i3+j*n, _mm_add_ps(Cij3, _mm_add_ps(_mm_mul_ps(Ai3k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai3k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai3k, Ajk), _mm_mul_ps(Ai3k1, Ajk1))))));
							// _mm_storeu_ps(C+i4+j*n, _mm_add_ps(Cij4, _mm_add_ps(_mm_mul_ps(Ai4k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai4k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai4k, Ajk), _mm_mul_ps(Ai4k1, Ajk1))))));
							// _mm_storeu_ps(C+i5+j*n, _mm_add_ps(Cij5, _mm_add_ps(_mm_mul_ps(Ai5k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai5k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai5k, Ajk), _mm_mul_ps(Ai5k1, Ajk1))))));
							// _mm_storeu_ps(C+i6+j*n, _mm_add_ps(Cij6, _mm_add_ps(_mm_mul_ps(Ai6k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai6k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai6k, Ajk), _mm_mul_ps(Ai6k1, Ajk1))))));
							// _mm_storeu_ps(C+i7+j*n, _mm_add_ps(Cij7, _mm_add_ps(_mm_mul_ps(Ai7k3, Ajk3), _mm_add_ps(_mm_mul_ps(Ai7k2, Ajk2), _mm_add_ps(_mm_mul_ps(Ai7k, Ajk), _mm_mul_ps(Ai7k1, Ajk1))))));
						}
					}
				}
			}
		}
	}
	//handle vertical edge case
	#pragma omp parallel for
	for (int j = 0; j < n; j++) {
		for (int k = 0; k < m; k+= HORIZONTAL_ROLL) {
			int k1 = k+1; 
			int k2 = k+2; 
			int k3 = k+3;
			// handle edge case here
			__m128 Ajk, Ajk1, Ajk2, Ajk3;
			Ajk = _mm_load1_ps(A+j*n1+k*n);
			if (k1 < m) {
				Ajk1 = _mm_load1_ps(A+j*n1+k1*n);
				if (k2 < m) {
					Ajk2 = _mm_load1_ps(A+j*n1+k2*n);
					if (k3 < m) {
						Ajk3 = _mm_load1_ps(A+j*n1+k3*n);
					}else {
						Ajk3 = _mm_setzero_ps();
					}
				}else {
					Ajk2 = _mm_setzero_ps();
					Ajk3 = _mm_setzero_ps();
				}
			}else {
				Ajk1 = _mm_setzero_ps();
				Ajk2 = _mm_setzero_ps();
				Ajk3 = _mm_setzero_ps();
			}
			for (int i = n/VERTICAL_ROLL*VERTICAL_ROLL; i < n; i++) {
				int i1 = i+4;
				__m128 Cij = _mm_load1_ps(C+i+j*n);
				__m128 Aik = _mm_load1_ps(A+i+k*n);
				__m128 Aik1 = _mm_load1_ps(A+i+k1*n);
				__m128 Aik2 = _mm_load1_ps(A+i+k2*n);
				__m128 Aik3 = _mm_load1_ps(A+i+k3*n);
				_mm_store_ss(C+i+j*n, _mm_add_ps(Cij, _mm_add_ps(_mm_mul_ps(Aik3, Ajk3), _mm_add_ps(_mm_mul_ps(Aik2, Ajk2), _mm_add_ps(_mm_mul_ps(Aik1, Ajk1), _mm_mul_ps(Aik, Ajk))))));
			}
		}
	}		

	
	 //    int i,j,k,k1,k2,j1,j2,jn,kn,n2, m3, n4, n1=n+1;
	 //    __m128 Ajk,Ajk1,Ajk2,Aj1k,Aj1k1,Aj1k2,Cij,Cij1,Aik,Aik1,Aik2,sumj,sumj1;
	 //    n2 = n/2*2;
	 //    n4 = n/4*4;
	 //    m3 = m/3*3;

		// for(k = 0; k < m3; k+=3){
		//     k1 = k+1; k2 = k+2;
		// 	for(j = 0; j < n2; j+=2){
		// 		j1 = j+1; j2 = j+2;
		// 		Ajk =  _mm_load1_ps(A+j*n1+k*n);
		// 		Ajk1 =  _mm_load1_ps(A+j*n1+(k1)*n);
		// 		Ajk2 = _mm_load1_ps(A+j*n1+(k2)*n);
	             			
		// 		Aj1k = _mm_load1_ps(A+j1*n1+(k)*n);
		// 		Aj1k1 = _mm_load1_ps(A+j1*n1+(k1)*n);
		// 		Aj1k2 = _mm_load1_ps(A+j1*n1+(k2)*n);

		// 		for(i = 0; i < n4; i+=4){
		// 			Cij = _mm_loadu_ps(C+i+j*n);
		// 			Cij1 = _mm_loadu_ps(C+i+j1*n);
		// 			Aik = _mm_loadu_ps(A+i+k*n);
		// 			Aik1 = _mm_loadu_ps(A+i+(k1)*n);
		// 			Aik2 = _mm_loadu_ps(A+i+(k2)*n);

		// 			sumj = _mm_add_ps(_mm_mul_ps(Ajk2, Aik2), _mm_add_ps(_mm_mul_ps(Ajk1, Aik1), _mm_add_ps(_mm_mul_ps(Ajk, Aik), Cij)));
		// 			sumj1 = _mm_add_ps(_mm_mul_ps(Aj1k2, Aik2), _mm_add_ps(_mm_mul_ps(Aj1k1, Aik1), _mm_add_ps(_mm_mul_ps(Aj1k, Aik), Cij1)));
					
		// 			_mm_storeu_ps(C+i+j*n, sumj);
		// 			_mm_storeu_ps(C+i+j1*n, sumj1);
		// 		}
		// 	}
		// }

		// for (i = n4; i < n; i++) {
		//     for(k = 0; k < m3; k+=3){
		// 	for(j = 0; j < n; j++) {
		// 	    C[i+j*n] = A[i+k*n] * A[j*n1+k*n] + C[i+j*n];
		// 	    C[i+j*n] = A[i+(k+1)*n] * A[j*n1+(k+1)*n] + C[i+j*n];			    
		// 	    C[i+j*n] = A[i+(k+2)*n] * A[j*n1+(k+2)*n] + C[i+j*n];
		// 	}
		//     } 
		// }


		// for(j = n2; j < n; j++){
		//     jn = j*n;
		//     for(k = 0; k < m3; k++){
		// 		kn = k*n;
		// 		Ajk =  _mm_load1_ps(A+j*n1+kn);
		// 		for(i = 0; i < n4; i+=4){
		// 	    	_mm_storeu_ps(C+i+jn, _mm_add_ps(_mm_mul_ps(Ajk, _mm_loadu_ps(A+i+kn)), _mm_loadu_ps(C+i+jn)));
		// 		}
		//     }
		// }	
  	
		// for(k = m3; k < m; k++){
		//     kn = k*n;
		// 	for(j = 0; j < n; j++){
		// 	    jn = j*n;
		// 	    Ajk =  _mm_load1_ps(A+j*n1+kn);
		// 		for(i = 0; i < n4; i+=4){
		// 			_mm_storeu_ps(C+i+jn, _mm_add_ps(_mm_mul_ps(Ajk, _mm_loadu_ps(A+i+kn)), _mm_loadu_ps(C+i+jn)));
		// 		}
		// 		for (i = n4; i < n; i++) {
		// 		    C[i+jn] =A[i+kn] * A[j*n1+kn] + C[i+jn];
		// 		}
		// 	}
		// }
}

