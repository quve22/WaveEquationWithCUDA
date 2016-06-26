#include "WaveEquationKernel.cuh"

#include "Surface.h"
#include <stdio.h>
#include <cuda_gl_interop.h>


__global__ void JacobiKernel(float4 *u0, float4 *u1, float4 *grid, int n, float a, float b)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;

	if ((c >= n) || (r >= n)) return;
	const int i = c + r * n;

	float ui_mw = 0;
	float ui_m1 = 0;
	float ui_p1 = 0;
	float ui_pw = 0;

	if (i - n >= 0)
		ui_mw = u1[i - n].y;
	if (i - 1 >= 0)
		ui_m1 = u1[i - 1].y;
	if (i + 1 < n * n)
		ui_p1 = u1[i + 1].y;
	if (i + n < n * n)
		ui_pw = u1[i + n].y;

	float ax = b * (ui_mw + ui_m1 + ui_p1 + ui_pw);
	float res = (2.0f * u1[i].y - u0[i].y - ax) / a;
	u0[i].y = u1[i].y;

	u1[i].y = res;

	const float gridRatioX = (float)blockDim.x / (float)n * blockIdx.x;
	const float gridRatioY = (float)blockDim.y / (float)n * blockIdx.y;

	grid[i].x = gridRatioX;
	grid[i].y = gridRatioY;
	grid[i].z = 0.0;
	grid[i].w = 1.0;
}


void kernelLauncher(float4 *u0, float4 *u1, float4 *grid, int n, float a, float b)
{
	// 한 블럭에 들어가는 thread의 수는 최대 1024개. 즉, 32 * 32까지가 한계라는 것.
#define BLOCK_SIZE 16

	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 gridSize = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

	static bool showFlag = false;
	if (!showFlag)
	{
		printf("Block Size : (%d, %d)\n", blockSize.x, blockSize.y);
		printf("Grid Size : (%d, %d)\n", gridSize.x, gridSize.y);
		
		showFlag = true;
	}
	
	JacobiKernel << <gridSize, blockSize >> >(u0, u1, grid, n, a, b);
}

//-----------------------

// GPU 내의 데이터에 바로 접근해서 계산을 하기 때문에
// wave buffer의 초기화가 끝난 후 실행이 되어야 한다.
cudaError_t initCudaDevice()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	printf("cudaSetDevice finished\n");

Error:
	return cudaStatus;
}