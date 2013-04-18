#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include <string.h>
#include <omp.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
	    int i,j,k,k1,k2,j1,j2,jn,kn,n2, m3, n4, n1=n+1;
	    __m128 Ajk,Ajk1,Ajk2,Aj1k,Aj1k1,Aj1k2,Cij,Cij1,Aik,Aik1,Aik2,sumj,sumj1;
	    n2 = n/2*2;
	    n4 = n/4*4;
	    m3 = m/3*3;

		for(k = 0; k < m3; k+=3){
		    k1 = k+1; k2 = k+2;
			for(j = 0; j < n2; j+=2){
				j1 = j+1; j2 = j+2;
				Ajk =  _mm_load1_ps(A+j*n1+k*n);
				Ajk1 =  _mm_load1_ps(A+j*n1+(k1)*n);
				Ajk2 = _mm_load1_ps(A+j*n1+(k2)*n);
	             			
				Aj1k = _mm_load1_ps(A+j1*n1+(k)*n);
				Aj1k1 = _mm_load1_ps(A+j1*n1+(k1)*n);
				Aj1k2 = _mm_load1_ps(A+j1*n1+(k2)*n);
				
				#pragma omp parallel for
					for(i = 0; i < n4; i+=4){
						Cij = _mm_loadu_ps(C+i+j*n);
						Cij1 = _mm_loadu_ps(C+i+j1*n);

						Aik = _mm_loadu_ps(A+i+k*n);
						Aik1 = _mm_loadu_ps(A+i+(k1)*n);
						Aik2 = _mm_loadu_ps(A+i+(k2)*n);

						sumj = _mm_add_ps(_mm_mul_ps(Ajk2, Aik2), _mm_add_ps(_mm_mul_ps(Ajk1, Aik1), _mm_add_ps(_mm_mul_ps(Ajk, Aik), Cij)));
						sumj1 = _mm_add_ps(_mm_mul_ps(Aj1k2, Aik2), _mm_add_ps(_mm_mul_ps(Aj1k1, Aik1), _mm_add_ps(_mm_mul_ps(Aj1k, Aik), Cij1)));
						
						_mm_storeu_ps(C+i+j*n, sumj);
						_mm_storeu_ps(C+i+j1*n, sumj1);
					}
			}
		}

		for (i = n4; i < n; i++) {
		    for(k = 0; k < m3; k+=3){
			for(j = 0; j < n; j++) {
			    C[i+j*n] = A[i+k*n] * A[j*n1+k*n] + C[i+j*n];
			    C[i+j*n] = A[i+(k+1)*n] * A[j*n1+(k+1)*n] + C[i+j*n];			    
			    C[i+j*n] = A[i+(k+2)*n] * A[j*n1+(k+2)*n] + C[i+j*n];
			}
		    } 
		}


		for(j = n2; j < n; j++){
		    jn = j*n;
		    for(k = 0; k < m3; k++){
				kn = k*n;
				Ajk =  _mm_load1_ps(A+j*n1+kn);
				for(i = 0; i < n4; i+=4){
			    	_mm_storeu_ps(C+i+jn, _mm_add_ps(_mm_mul_ps(Ajk, _mm_loadu_ps(A+i+kn)), _mm_loadu_ps(C+i+jn)));
				}
		    }
		}	
  	
		for(k = m3; k < m; k++){
		    kn = k*n;
			for(j = 0; j < n; j++){
			    jn = j*n;
			    Ajk =  _mm_load1_ps(A+j*n1+kn);
				for(i = 0; i < n4; i+=4){
					_mm_storeu_ps(C+i+jn, _mm_add_ps(_mm_mul_ps(Ajk, _mm_loadu_ps(A+i+kn)), _mm_loadu_ps(C+i+jn)));
				}
				for (i = n4; i < n; i++) {
				    C[i+jn] =A[i+kn] * A[j*n1+kn] + C[i+jn];
				}
			}
		}
}