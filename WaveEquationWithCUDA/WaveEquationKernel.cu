#include "WaveEquationKernel.cuh"

#include "Surface.h"
#include <stdio.h>
#include <cuda_gl_interop.h>

float4 *cuda_data = NULL;

extern "C" void map_texture(cudaGraphicsResource *resource, int w, int h)
{
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void **)(&cuda_data), &size, resource);
}


cudaGraphicsResource *resources[1];

__global__ void JacobiKernel()
{
	
}

// GPU 내의 데이터에 바로 접근해서 계산을 하기 때문에
// wave buffer의 초기화가 끝난 후 실행이 되어야 한다.
cudaError_t initCudaDevice()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	printf("cudaSetDevice finished\n");

Error:

	return cudaStatus;
}

void set_cuda_ogl_interoperability()
{
	GLuint pbo;

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, GRID * GRID * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cudaGLSetGLDevice(0);

	// Register Pixel Buffer Object as CUDA graphics resource.
	cudaGraphicsGLRegisterBuffer(resources, pbo, cudaGraphicsMapFlagsNone);

	cudaStream_t cuda_stream;
	cudaStreamCreate(&cuda_stream);

	// 그래픽 리소스를 CUDA 스트림에 맵핑.
	cudaGraphicsMapResources(1, resources, cuda_stream);

	// CUDA Function 호출.
	map_texture(resources[0], GRID, GRID);

	cudaGraphicsUnmapResources(1, resources, cuda_stream);
	cudaStreamDestroy(cuda_stream);
}

cudaError_t calculateWaveEquation()
{
	dim3 blockSize(16, 16);
	dim3 gridSize(GRID / blockSize.x, GRID / blockSize.y);

	cudaStream_t cuda_stream;
	cudaError_t cudaStatus;
	// printf("calculateWaveEquation() started\n");

	cudaStreamCreate(&cuda_stream);
	cudaGraphicsMapResources(1, resources, cuda_stream);

	JacobiKernel <<< 16, 16 >>>();

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaGraphicsUnmapResources(1, resources, cuda_stream);
	cudaStreamDestroy(cuda_stream);
	// printf("calculateWaveEquation() finished\n");
	return cudaStatus;
}