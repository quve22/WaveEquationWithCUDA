#include <stdio.h>

#include "GLDrawing.h"
#include "GLInput.h"
#include "Surface.h"

#include "Shaders\LoadShaders.h"

#include "WaveEquationKernel.cuh"

#define RADIAN 1.7f
#define TO_RADIAN 0.01745329252f  
#define TO_DEGREE 57.295779513f

GLuint h_ShaderProgram_simple;
GLuint h_ShaderProgram_compute;
GLuint h_ShaderProgram_compute_ax;
GLuint h_ShaderProgram_compute_normal;

GLuint loc_ModelViewProjectionMatrix_simple, loc_primitive_color;


glm::mat4 ModelViewProjectionMatrix, ModelViewMatrix;
glm::mat4 ViewMatrix, ProjectionMatrix;

/* process menu option 'op' */
void menu(int op)
{
	switch(op) {
		case 'Q':
		case 'q':
			exit(0);
	}
}

/* reshaped window */
void reshape(int width, int height)
{
	float aspect_ratio;
	glViewport(0, 0, width, height);

	aspect_ratio = (float)width / height;
	ProjectionMatrix = glm::perspective(40.0f * TO_RADIAN, aspect_ratio, 0.1f, 500.0f);

	glutPostRedisplay();
}

/* render the scene */
void display()
{
	// Wave Equation CPU 연산.
#ifdef USE_GPU
	const int local_size = 16;

	// Compute Wave equation.
	for (int i = 0; i < ITER_COUNT; i++) {
		glUseProgram(h_ShaderProgram_compute_ax);
		glDispatchCompute(GRID / local_size, GRID / local_size, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		glUseProgram(h_ShaderProgram_compute);
		glDispatchCompute(GRID / local_size, GRID / local_size, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	// Compute the normals
	glUseProgram(h_ShaderProgram_compute_normal);
	glDispatchCompute(GRID / local_size, GRID / local_size, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
#else
	glBindVertexArray(waveVao);
	glBindBuffer(GL_ARRAY_BUFFER, u1Bufs);
	float *posBufs = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	if (posBufs)
	{
		for (int i = 0; i < GRID; i++)
		{
			for (int j = 0; j < GRID; j++)
			{
				posBufs[(i * GRID + j) * 4 + 1] += 0.01f; 
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
	}
	glBindVertexArray(0);
#endif

	calculateWaveEquation();

	glUseProgram(h_ShaderProgram_simple);
	glClear(GL_COLOR_BUFFER_BIT);

	// 표면 그리기.
	ModelViewProjectionMatrix = ProjectionMatrix * ModelViewMatrix;
	drawSurface(h_ShaderProgram_simple, ModelViewProjectionMatrix);

	glutSwapBuffers();
}

void timerScene(int timestamp_scene)
{
	glutPostRedisplay();
	glutTimerFunc(40, timerScene, 1);
}

void cleanup(void)
{

}

/* initialize OpenGL settings */
void initGL(int width, int height)
{
	reshape(width, height);

	glClearColor(0.5f, 0.5f, 0.5f, 0.1f);	
	glClearDepth(1.0f);

	// glEnable(GL_DEPTH_TEST);
}

void initGLEW()
{
	GLenum error;

	glewExperimental = GL_TRUE;

	error = glewInit();
	if(error != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(error));
		exit(-1);
	}
	fprintf(stdout, "*********************************************************\n");
	fprintf(stdout, " - GLEW version supported: %s\n", glewGetString(GLEW_VERSION));
	fprintf(stdout, " - OpenGL renderer: %s\n", glGetString(GL_RENDERER));
	fprintf(stdout, " - OpenGL version supported: %s\n", glGetString(GL_VERSION));
	fprintf(stdout, "*********************************************************\n\n");
}

void prepareShaderProgram(void)
{
	// Compute Shader 컴파일 및 링크.
	ShaderInfo shader_info_compute[2] = {
		{ GL_COMPUTE_SHADER, "Shaders/wave.comp" }, 
		{ GL_NONE, NULL }
	};
	h_ShaderProgram_compute = LoadShaders(shader_info_compute);
	
	glUseProgram(h_ShaderProgram_compute);
	glUniform1f(glGetUniformLocation(h_ShaderProgram_compute, "a"), mWave.alpha);

	ShaderInfo shader_info_compute_ax[2] = {
		{ GL_COMPUTE_SHADER, "Shaders/wave_ax.comp" }, 
		{ GL_NONE, NULL }
	};
	h_ShaderProgram_compute_ax = LoadShaders(shader_info_compute_ax);

	glUseProgram(h_ShaderProgram_compute_ax);
	glUniform1f(glGetUniformLocation(h_ShaderProgram_compute_ax, "b"), mWave.beta);

	ShaderInfo shader_info_compute_normal[2] = {
		{ GL_COMPUTE_SHADER, "Shaders/wave_normal.comp" }, 
		{ GL_NONE, NULL }
	};
	h_ShaderProgram_compute_normal = LoadShaders(shader_info_compute_normal);


	// 기본 쉐이더 컴파일 및 링크.
	ShaderInfo shader_info_simple[3] = {
		{ GL_VERTEX_SHADER, "Shaders/simple.vert" },
		{ GL_FRAGMENT_SHADER, "Shaders/simple.frag" },
		{ GL_NONE, NULL }
	};

	h_ShaderProgram_simple = LoadShaders(shader_info_simple);

	glUseProgram(h_ShaderProgram_simple);
	loc_ModelViewProjectionMatrix_simple = glGetUniformLocation(h_ShaderProgram_simple, "ModelViewProjectionMatrix");
}

void prepareScene()
{
	// prepareSurface(N_WIDTH);
	initWaveBuffers(GRID);

	ViewMatrix = lookAt(glm::vec3(7.0, 3.0, 7.0), glm::vec3(0, 0, 0), glm::vec3(0.0, 1.0, 0.0));
	ModelViewMatrix = ViewMatrix;

	glutSwapBuffers();
}

void InitGLUTSetting(int argc, char** argv)
{
	// CUDA 설정.
	initCudaDevice();

	// GLUT 설정.
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutInitContextVersion(4, 3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow("Wave Equation with CUDA");

	glClearColor(0.5f, 0.5f, 0.5f, 0.1f);

	initGL(800, 800);
	initGLEW();

	// register glut call backs
	glutKeyboardFunc(keyboardDown);
	glutKeyboardUpFunc(keyboardUp);
	glutSpecialFunc(keyboardSpecialDown);
	glutSpecialUpFunc(keyboardSpecialUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutTimerFunc(40, timerScene, 0);
	glutCloseFunc(cleanup);

	// create a sub menu 
	int subMenu = glutCreateMenu(menu);
	glutAddMenuEntry("Do nothing", 0);
	glutAddMenuEntry("Really Quit", 'q');

	// create main "right click" menu
	glutCreateMenu(menu);
	glutAddSubMenu("Sub Menu", subMenu);
	glutAddMenuEntry("Quit", 'q');
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	// create shader program.
	prepareShaderProgram();

	prepareScene();

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();
}