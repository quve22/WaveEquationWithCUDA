#include "GLInput.h"
#include <stdio.h>

double xdeg;
double ydeg;
bool g_display2DMode = false;

/* executed when a regular key is pressed */
void keyboardDown(unsigned char key, int x, int y) {

	switch(key) {
		case 't':
		case 'T':
			g_display2DMode = !g_display2DMode;
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

bool g_isClicked = false;
void mouseClick(int button, int state, int x, int y)
{
	if(state == GLUT_LEFT_BUTTON)
	{


		g_isClicked = true;
	}
	else
	{
		g_isClicked = false;
	}
}

/* executed when the mouse moves to position ('x', 'y') */
void mouseMotion(int x, int y)
{
	if(g_isClicked)
	{
		static double prev_xpos = x, prev_ypos = y;

		ydeg += (x - prev_xpos);
		xdeg += (y - prev_ypos);

		prev_xpos = x;
		prev_ypos = y;
	}
}