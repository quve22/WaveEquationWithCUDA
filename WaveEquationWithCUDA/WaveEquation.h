#include <stdio.h>
#include <stdlib.h>

/*********************************************************

	N x N 공간에 대한 Wave Equation을 계산할 시
	모든 size들은 N-2 x N-2 공간에 대한 계산을 수행한다.
	경계 부분의 U 값들은 모두 0이기 때문에 계산할 필요 없음.

**********************************************************/

// #define USE_GPU

class Wave {
private:
	float *X;
	float *U_1;		// 현재의 높이값.
	float *U_0;		// 이전의 높이값.
	float *dU;
	float **A;
	float *B;

	int size;			// wave의 가로 한 변 길이.
	int size2;			// size^2
	float deltaT;

	void initU();
	void initDeltaU();
	void calculateInitialU_1();
	void initMatrixA();
	void initVectorB();
	void calculateVectorB();

	float accum_t;
public:
	Wave(int _size /*이 값은 실제 공간의 N값 N-2값이 아니다.*/, float _alpha, float _deltaT, float _h)
	{
		size = _size;
		size2 = size * size;
		beta = - (_alpha * _alpha) * (_deltaT * _deltaT) / _h;
		alpha = 1 - 4*beta;
		deltaT = _deltaT;

		initU();
		initDeltaU();
		calculateInitialU_1();
		//initMatrixA();
		initVectorB();
	}

	float alpha;
	float beta;

	float* getU0(){ return U_0; };
	float* getU1(){ return U_1; };
	float** getA(){ return A; };
	void calculateVectorU_1();	// 매 루프마다 이를 실행.
};