#include "GLDrawing.h"

extern GLuint surface_VAO;

void prepareSurface(int N);
void draw_surface(GLuint hProgramId, glm::mat4 mvpMatrix);