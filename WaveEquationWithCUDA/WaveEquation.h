#include <stdio.h>
#include <stdlib.h>

/*********************************************************

	N x N ������ ���� Wave Equation�� ����� ��
	��� size���� N-2 x N-2 ������ ���� ����� �����Ѵ�.
	��� �κ��� U ������ ��� 0�̱� ������ ����� �ʿ� ����.

**********************************************************/

// #define USE_GPU

class Wave {
private:
	float *X;
	float *U_1;		// ������ ���̰�.
	float *U_0;		// ������ ���̰�.
	float *dU;
	float **A;
	float *B;

	int size;			// wave�� ���� �� �� ����.
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
	Wave(int _size /*�� ���� ���� ������ N�� N-2���� �ƴϴ�.*/, float _alpha, float _deltaT, float _h)
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
	void calculateVectorU_1();	// �� �������� �̸� ����.
};