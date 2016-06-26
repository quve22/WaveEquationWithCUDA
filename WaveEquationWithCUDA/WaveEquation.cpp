#include <stdio.h>
#include <string.h>
#include <math.h>

#include "WaveEquation.h"

void Wave::initU()
{
	X = (float *) malloc (sizeof(float) * size2);
	U_0 = (float *) malloc (sizeof(float) * size2);
	for(int i=0 ; i<size2 ; i++)
		U_0[i] = 0.0;

	memcpy(X, U_0, sizeof(float) * size2);

	U_1 = (float *) malloc (sizeof(float) * size2);
	for(int i=0 ; i<size2 ; i++)
		U_1[i] = 0.0;
}

void Wave::initDeltaU()
{
	dU = (float *) malloc (sizeof(float) * size2);
	for(int i=0 ; i<size2 ; i++)
		dU[i] = 0.0;

	// 초기 속력을 설정.
	// dU[size2/2] = -1.0f;
	const float ispeed = 3.0f;
	/*
	dU[size * (size/2 - 1) + size/2] = ispeed;
	dU[size * (size/2 - 1) + size/2 + 1] = ispeed/2.0f;
	dU[size * (size/2 - 1) + size/2 - 1] = ispeed/2.0f;
	dU[size * (size/2 - 2) + size/2] = ispeed/2.0f;
	dU[size * (size/2 - 0) + size/2] = ispeed/2.0f;
	*/
	
	dU[size * (size / 3 - 1) + size / 3] = ispeed;
	dU[(size * (size / 3 - 1) + size / 3) * 2] = ispeed;
	
}

void Wave::calculateInitialU_1()
{
	for(int i=0 ; i<size2 ; i++)
	{
		U_1[i] = U_0[i] + deltaT * dU[i];
	}
}

void Wave::initMatrixA()
{
	A = (float**) malloc (sizeof(float*) * size2);
	for(int i=0 ; i<size2 ; i++)
		A[i] = (float *) malloc (sizeof(float) * size2);

	for(int i=0 ; i<size2 ; i++)
		for(int j=0 ; j<size2 ; j++)
			A[i][j] = 0.0;

	for(int i=0 ; i<size2 ; i++)
	{
		A[i][i] = alpha;

		if(i-1 >= 0)
			A[i][i-1] = beta;
		if(i+1 <= size2-1)
			A[i][i+1] = beta;
		if(i-size >= 0)
			A[i][i-size] = beta;
		if(i+size <= size2-1)
			A[i][i+size] = beta;
	}

	for(int i=1 ; i<=size-1 ; i++)
	{
		int idx = size * i;
		A[idx - 1][idx] = A[idx][idx - 1] = 0;
	}

	// Print A matrix
	/*
	for(int i=0 ; i<size2 ; i++)
	{
		for(int j=0 ; j<size2 ; j++)
		{
			printf("%.2f\t", A[i][j]);
		}
		printf("\n");
	}
	*/
}

void Wave::initVectorB()
{
	B = (float*) malloc (sizeof(float) * size2);

	calculateVectorB();
}

void Wave::calculateVectorB()
{
	for(int i=0 ; i<size2 ; i++)
	{
		B[i] = 2 * U_1[i] - U_0[i];
	}
}

// 최적화가 전혀 수행되지 않은 코드.
void Wave::calculateVectorU_1()
{
	// Jacobi
	for(int i=0 ; i<size2 ; i++)
	{
		float ax = 0;
		for(int j=0 ; j<size2 ; j++)
		{
			if(i == j) continue;
			ax += A[i][j] * U_1[j];
		}
		X[i] = (B[i] - ax) / A[i][i];
	}

	// Gauss-Seidel
	/*
	for(int i=0 ; i<size2 ; i++)
	{
		float ax = 0, bx = 0;
		for(int j=0 ; j<i ; j++)
		{
			ax += (A[i][j] * X[j]);
		}
		for(int j=size2-1 ; j>i ; j--)
		{
			bx += (A[i][j] * U_1[j]);
		}
		X[i] = (-ax - bx + B[i]) * A[i][i];
	}
	*/

	// U_1에 있던 값을 U_0으로, X에 계산 된 값을 U_1로 이동.
	memcpy(U_0, U_1, sizeof(float) * size2);
	memcpy(U_1, X, sizeof(float) * size2);
	
	const float scale = 0.3f;
	accum_t+=1;

	float waveValue = scale * sinf(accum_t * 0.1f);
	
	U_1[size * (size/2 - 0) + size/2] = waveValue;
	U_1[size * (size/2 - 1) + size/2 - 1] = waveValue;
	U_1[size * (size/2 - 1) + size/2] = waveValue;
	U_1[size * (size/2 - 1) + size/2 + 1] = waveValue;
	U_1[size * (size/2 - 2) + size/2] = waveValue;
	
	// 새로운 값으로 B 벡터를 갱신.
	calculateVectorB();
}