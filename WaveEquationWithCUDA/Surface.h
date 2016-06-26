#include "WaveEquation.h"
#include "GLDrawing.h"

#define GRID 64

//#define USE_GPU
#define USE_CUDA
#define ITER_COUNT  50

#define al			0.5
#define DELTH_T	    0.1 / ITER_COUNT
#define dh			1.0

#ifdef USE_GPU
extern Wave mWave;
#elif defined USE_CUDA
extern Wave mWave;
#else
extern Wave mWave;
#endif


extern GLuint surface_VAO;

extern GLuint u0Bufs, u1Bufs, axBuf, gridBuf;
extern GLuint waveVao;
extern GLuint g_pboTexture;

extern struct cudaGraphicsResource *cuda_u0_resource;
extern struct cudaGraphicsResource *cuda_u1_resource;
extern struct cudaGraphicsResource *cuda_ax_resource;
extern struct cudaGraphicsResource *cuda_grid_resource;

void initWaveBuffers(int n);
void prepareSurface(int N);

void drawSurface(GLuint hProgramId, glm::mat4 mvpMatrix);