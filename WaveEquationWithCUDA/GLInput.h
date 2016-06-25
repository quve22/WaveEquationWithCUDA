#ifndef __GL_INPUT_H__
#define __GL_INPUT_H__

#include <GL/glew.h>
#include <GL/freeglut.h>

extern double xdeg;
extern double ydeg;
extern bool g_display2DMode;

void keyboardDown(unsigned char key, int x, int y);
void keyboardUp(unsigned char key, int x, int y);
void keyboardSpecialDown(int k, int x, int y);
void keyboardSpecialUp(int k, int x, int y);
void mouseClick(int button, int state, int x, int y);
void mouseMotion(int x, int y);

#endif //__GL_INPUT_H__