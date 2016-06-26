#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t initCudaDevice();

void kernelLauncher(float4 *u0, float4 *u1, float4 *grid, int n, float a, float b);