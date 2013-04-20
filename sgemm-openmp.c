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
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			for (int k = 0; k < m; k++) {
				
			}
		}
	}
}

