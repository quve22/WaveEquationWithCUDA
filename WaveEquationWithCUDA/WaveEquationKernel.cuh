#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern cudaGraphicsResource *resources[1];

cudaError_t initCudaDevice();
cudaError_t calculateWaveEquation();

void set_cuda_ogl_interoperability();