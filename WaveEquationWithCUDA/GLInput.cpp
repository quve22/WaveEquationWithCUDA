#include "GLInput.h"
#include "Surface.h"
#include "GLDrawing.h"
#include <stdio.h>

double xdeg;
double ydeg;
bool g_display2DMode = false;

void resetWave()
{
	glBindBuffer(GL_ARRAY_BUFFER, u0Bufs);
	float *posBufs = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	if (posBufs)
	{
		for (int i = 0; i < GRID; i++)
		{
			for (int j = 0; j < GRID; j++)
			{
				posBufs[(i * GRID + j) * 4 + 1] = 0.0f;
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
	}

	glBindBuffer(GL_ARRAY_BUFFER, u1Bufs);
	posBufs = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	if (posBufs)
	{
		for (int i = 0; i < GRID; i++)
		{
			for (int j = 0; j < GRID; j++)
			{
				posBufs[(i * GRID + j) * 4 + 1] = 0.0f;
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
	}
}

/* executed when a regular key is pressed */
void keyboardDown(unsigned char key, int x, int y) {

	switch(key) {
		case 't':
		case 'T':
			g_display2DMode = !g_display2DMode;
			break;
		case 'r':
		case 'R':
			resetWave();
			break;
		case 'Q':
		case 'q':
		case  27:   // ESC
			exit(0);
	}
}

/* executed when a regular key is released */
void keyboardUp(unsigned char key, int x, int y) {

}

/* executed when a special key is pressed */
void keyboardSpecialDown(int k, int x, int y) {

}

/* executed when a special key is released */
void keyboardSpecialUp(int k, int x, int y) {

}

double prev_xpos, prev_ypos;
bool g_isClicked = false;
void mouseClick(int button, int state, int x, int y)
{
	if(state == GLUT_LEFT_BUTTON)
	{
		g_isClicked = true;

		if (g_display2DMode)
		{
			float* ptr;
			float* pixels; /* the downloaded pixels. */
			int nbytes; /* number of bytes in the pbo buffer. */

			nbytes = GRID * GRID * sizeof(GLfloat)* 4;
			pixels = (float *)malloc(nbytes);

			glBindBuffer(GL_PIXEL_PACK_BUFFER, u1Bufs);
			ptr = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_WRITE);
			if (NULL != ptr) {
				x = (int)((float)x / (float)W * (float)GRID);
				y = (int)((float)y / (float)H * (float)GRID);

				ptr[(y * GRID + x) * 4 + 1] += 0.001f;

				glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
			}

			free(pixels);
		}
	}
	else
	{
		g_isClicked = false;
	}

	prev_xpos = x;
	prev_ypos = y;
}

/* executed when the mouse moves to position ('x', 'y') */
void mouseMotion(int x, int y)
{
	if(g_isClicked)
	{
		ydeg += (x - prev_xpos);
		xdeg += (y - prev_ypos);

		prev_xpos = x;
		prev_ypos = y;
	}
}