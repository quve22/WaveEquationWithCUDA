#include <stdio.h>

#include "GLDrawing.h"
#include "GLInput.h"
#include "Surface.h"

#include "Shaders\LoadShaders.h"

#include "WaveEquationKernel.cuh"

#define RADIAN 1.7f
#define TO_RADIAN 0.01745329252f  
#define TO_DEGREE 57.295779513f

static int W, H;

GLuint h_ShaderProgram_simple;
GLuint h_ShaderProgram_compute;
GLuint h_ShaderProgram_compute_ax;
GLuint h_ShaderProgram_compute_normal;

GLuint h_ShaderProgram_quad;

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

	W = width;
	H = height;

	glutPostRedisplay();
}

void cuda_calculation()
{
	float4 *u0_out = 0;
	float4 *u1_out = 0;
	float4 *ax_out = 0;

	cudaGraphicsMapResources(1, &cuda_u0_resource, 0);
	cudaGraphicsMapResources(1, &cuda_u1_resource, 0);
	cudaGraphicsMapResources(1, &cuda_ax_resource, 0);

	cudaGraphicsResourceGetMappedPointer((void **)&u0_out, NULL, cuda_u0_resource);
	cudaGraphicsResourceGetMappedPointer((void **)&u1_out, NULL, cuda_u1_resource);
	cudaGraphicsResourceGetMappedPointer((void **)&ax_out, NULL, cuda_ax_resource);

	kernelLauncher(u0_out, u1_out, GRID, mWave.alpha, mWave.beta);

	cudaGraphicsUnmapResources(1, &cuda_u0_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_u1_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_ax_resource, 0);
}

void checkGLError()
{
	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("Error occured. glError:0x%04X\n", err);
	}
}


GLuint g_pointVBO;
GLuint g_pointVAO;

GLfloat g_vertexBuffer[9] = {
	0.0f, 0.622008459f, 0.0f,      // top
	-0.5f, -0.311004243f, 0.0f,     // bottom left
	0.5f, -0.311004243f, 0.0f       // bottom right
};
GLfloat g_color[4] = { 0.63671875f, 0.76953125f, 0.22265625f, 1.0f };

float g_pointCoord[4] = { 0.0, 0.0, 0.0, 1.0 };

GLuint g_pboTexture;

void initPoint()
{
	glGenBuffers(1, &g_pointVBO);

	glBindBuffer(GL_ARRAY_BUFFER, g_pointVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_pointCoord), &g_pointCoord[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenVertexArrays(1, &g_pointVAO);

	glBindVertexArray(g_pointVAO);
	glBindBuffer(GL_ARRAY_BUFFER, g_pointVBO);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Step 1. 데이터를 2D로 표현할 텍스쳐를 생성.
	glGenTextures(1, &g_pboTexture);

	// Step 2. 텍스쳐 데이터를 저장할 메모리 공간을 우선 생성.
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, g_pboTexture);
	// Step 2-1. 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GRID, GRID, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void readPixelBufferObject(GLuint pboId)
{
	float* ptr;
	float* pixels; /* the downloaded pixels. */
	int nbytes; /* number of bytes in the pbo buffer. */

	nbytes = GRID * GRID * sizeof(GLfloat)* 4;
	pixels = (float *)malloc(nbytes);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, u1Bufs);
	ptr = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	if(NULL != ptr) {
		memcpy(pixels, ptr, nbytes);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	}

	free(pixels);
}

void drawPoint()
{
	glBindVertexArray(g_pointVAO);

	readPixelBufferObject(u1Bufs);

	// Step 3. u1Bufs에 해당하는 메모리 공간에 있는 데이터를 시각화 할 예정. 이 Pixel Buffer를 바인딩 한다.
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, u1Bufs);
	// Step 4. 데이터를 담을 텍스쳐를 바인딩 한다.
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, g_pboTexture);

	// Step 5. 사용할 텍스쳐를 활성화 하고 glTexSubImage2D를 이용해 pixel buffer의 데이터를 텍스쳐가 가리키도록 한다.
	// glTexSubImage2D는 가장 마지막 매개변수가 0일 경우, 현재 GL_PIXEL_UNPACK_BUFFER에 바인딩이 되어 있는 데이터를 텍스쳐의 데이터로서 사용한다.
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID, GRID, GL_RGBA, GL_FLOAT, (void *)0);

	// Step 6. 위 과정을 마치면 쉐이더의 sampler2D를 통해 텍스쳐 데이터에 접근이 가능.
	// 아래와 같이 계속 glGetUniformLocation을 호출하는 것이 좋은 구조는 아니지만 지금은 구현의 편의를 위해 사용.
	glUniform1i(glGetUniformLocation(h_ShaderProgram_quad, "u_texture"), g_pboTexture);
	glDrawArrays(GL_POINTS, 0, 1);

	// Step 7. 사용이 종료됐으니 바인딩 해제.
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindVertexArray(0);
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
#elif defined USE_CUDA
	for(int i = 0; i < ITER_COUNT; i++)
		cuda_calculation();

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

	// 이전 버전 CUDA 작업.
	// calculateWaveEquation();

	if(!g_display2DMode)
	{
		glClearColor(0.5f, 0.5f, 0.5f, 0.1f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(h_ShaderProgram_simple);
		glClear(GL_COLOR_BUFFER_BIT);

		// 표면 그리기.
		glm::mat4 transf = glm::mat4(1.0f);
		transf = glm::rotate(transf, glm::radians((float)ydeg), glm::vec3(0.0f, 1.0f, 0.0f));
		transf = glm::rotate(transf, glm::radians((float)xdeg), glm::vec3(1.0f, 0.0f, 0.0f));
		
		ModelViewMatrix = ViewMatrix * transf;
		ModelViewProjectionMatrix = ProjectionMatrix * ModelViewMatrix;
		drawSurface(h_ShaderProgram_simple, ModelViewProjectionMatrix);

		glUseProgram(0);
	}
	else
	{
		glClearColor(0.5f, 0.5f, 0.5f, 0.1f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(h_ShaderProgram_quad);

		drawPoint();

		glUseProgram(0);
	}

	// fps 계산.
	static int nbFrames = 0;
	static double lastTime = glutGet(GLUT_ELAPSED_TIME);
	double currTime = glutGet(GLUT_ELAPSED_TIME);

	nbFrames++;
	if(currTime - lastTime >= 1000.0)
	{
		printf("%f ms / frame\n", 1000.0 / double(nbFrames));
		nbFrames = 0;
		lastTime = currTime;
	}

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

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
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
	// 2D 평면 그리는 쉐이더 컴파일.
	ShaderInfo shader_info_quad[3] = {
		{ GL_VERTEX_SHADER, "Shaders/quad.vert" },
		{ GL_FRAGMENT_SHADER, "Shaders/quad.frag" },
		{ GL_NONE, NULL }
	};
	h_ShaderProgram_quad = LoadShaders(shader_info_quad);
	glUseProgram(h_ShaderProgram_quad);
	GLuint loc_pointSize = glGetUniformLocation(h_ShaderProgram_quad, "u_pointSize");
	glUniform1f(loc_pointSize, W);


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
	initPoint();
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