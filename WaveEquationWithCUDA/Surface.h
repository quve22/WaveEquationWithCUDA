#include "WaveEquation.h"
#include "GLDrawing.h"

#define GRID 80

#define USE_GPU
#define ITER_COUNT  100

#define al			1.0
#define DELTH_T	    0.1 / ITER_COUNT
#define dh			1.0

#ifdef USE_GPU
extern Wave mWave;
#else
extern Wave mWave(GRID, 1.0, 0.1, 1.0);
#endif


extern GLuint surface_VAO;

extern GLuint u0Bufs, u1Bufs;
extern GLuint waveVao;

void initWaveBuffers(int n);
void prepareSurface(int N);

void draw_surface(GLuint hProgramId, glm::mat4 mvpMatrix);
void drawSurface(GLuint hProgramId, glm::mat4 mvpMatrix);