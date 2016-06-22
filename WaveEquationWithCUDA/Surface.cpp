#include "Surface.h"
#include "GLInput.h"

#include <vector>
#include <FreeImage/FreeImage.h>

#define PRIM_RESTART 0xffffff

struct coord {
	float x;
	float y;
	float z;
};
GLuint surface_VBO, surface_VAO;
int surface_n_triangles;
coord *surface_vert;
int *surface_vert_indices;

coord *normal_vert;

glm::vec2 waveSize(5.0f, 5.0f);

GLuint readBuf;
GLuint posBufs[2], velBufs[2];
GLuint u0Bufs, u1Bufs;
GLuint aBufs, invaBufs;
GLuint normBuf, elBuf, tcBuf;
GLuint axBuf;

GLuint waveVao;
GLuint numElements;
GLuint textureId;

Wave mWave = Wave(GRID, al, DELTH_T, dh);

void My_glTexImage2D_from_file(char *filename) {
	FREE_IMAGE_FORMAT tx_file_format;
	int tx_bits_per_pixel;
	FIBITMAP *tx_pixmap, *tx_pixmap_32;

	int width, height;
	GLvoid *data;

	tx_file_format = FreeImage_GetFileType(filename, 0);
	// assume everything is fine with reading texture from file: no error checking
	tx_pixmap = FreeImage_Load(tx_file_format, filename);
	tx_bits_per_pixel = FreeImage_GetBPP(tx_pixmap);

	fprintf(stdout, " * A %d-bit texture was read from %s.\n", tx_bits_per_pixel, filename);
	if(tx_bits_per_pixel == 32)
		tx_pixmap_32 = tx_pixmap;
	else {
		fprintf(stdout, " * Converting texture from %d bits to 32 bits...\n", tx_bits_per_pixel);
		tx_pixmap_32 = FreeImage_ConvertTo32Bits(tx_pixmap);
	}

	width = FreeImage_GetWidth(tx_pixmap_32);
	height = FreeImage_GetHeight(tx_pixmap_32);
	data = FreeImage_GetBits(tx_pixmap_32);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data);
	// Allocate storage
	// glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
	// Copy data into storage
	// glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
	fprintf(stdout, " * Loaded %dx%d RGBA texture into graphics memory.\n\n", width, height);

	FreeImage_Unload(tx_pixmap_32);
	if(tx_bits_per_pixel != 32)
		FreeImage_Unload(tx_pixmap);
}

void initWaveBuffers(int n)
{
	glm::ivec2 nParticles(n, n);

	// �� �ε��� ���� �������� ������Ƽ�긦 �ٽ� �׸���. GL_TRIANGLE_STRIP ��� ����.
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(PRIM_RESTART);

	// Initial transform
	glm::mat4 transf = glm::mat4(1.0f);
	// transf = glm::rotate(transf, glm::radians(ydeg), glm::vec3(0.0f, 1.0f, 0.0f));
	// transf = glm::rotate(transf, glm::radians(xdeg), glm::vec3(1.0f, 0.0f, 0.0f));
	transf = glm::translate(transf, glm::vec3(-waveSize.x / 2, 0, waveSize.y / 2));

	float* U0 = ::mWave.getU0();
	float* U1 = ::mWave.getU1();

	// Initial positions of the particles
	std::vector<GLfloat> initU0, initU1;
	std::vector<GLfloat> ad, invad;	//A diagonal
	std::vector<GLfloat> axValue;

	std::vector<float> initTc;

	float dx = waveSize.x / (nParticles.x - 1);
	float dy = waveSize.y / (nParticles.y - 1);
	float ds = 1.0f / (nParticles.x - 1);
	float dt = 1.0f / (nParticles.y - 1);
	glm::vec4 p(0.0f, 0.0f, 0.0f, 1.0f);
	for(int i = 0; i < nParticles.y; i++) {
		for(int j = 0; j < nParticles.x; j++) {
			float initU0Pos = 0.0f;
			float initU1Pos = 0.0f;

			p.x = dx * j;
			p.y = U0[i*nParticles.x + j];
			p.y = initU0Pos;
			p.z = -dy * i;
			p = transf * p;
			initU0.push_back(p.x);
			initU0.push_back(U0[i*nParticles.x + j]);
			//initU0.push_back(initU0Pos);
			initU0.push_back(p.z);
			initU0.push_back(1.0f);

			initU1.push_back(p.x);
			initU1.push_back(U1[i*nParticles.x + j]);
			//initU1.push_back(initU1Pos);
			initU1.push_back(p.z);
			initU1.push_back(1.0f);

			initTc.push_back(ds * j);
			initTc.push_back(dt * i);
		}
	}

	for(int i = 0; i < nParticles.x * nParticles.y; i++)
	{
		ad.push_back(1);
		invad.push_back(1.0f / 1);
		axValue.push_back(0);
	}

	// Every row is one triangle strip
	std::vector<GLuint> el;
	for(int row = 0; row < nParticles.y - 1; row++) {
		for(int col = 0; col < nParticles.x; col++) {
			el.push_back((row + 1) * nParticles.x + (col));
			el.push_back((row)* nParticles.x + (col));
		}
		el.push_back(PRIM_RESTART);
	}

	// We need buffers for position (2), element index,
	// velocity (2), normal, and texture coordinates.
	GLuint bufs[8];
	glGenBuffers(8, bufs);
	u0Bufs = bufs[0];
	u1Bufs = bufs[1];
	aBufs = bufs[2];
	invaBufs = bufs[3];
	normBuf = bufs[4];
	elBuf = bufs[5];
	tcBuf = bufs[6];
	axBuf = bufs[7];

	GLuint parts = nParticles.x * nParticles.y;

	// Position�� ���� ����.
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, u0Bufs);
	glBufferData(GL_SHADER_STORAGE_BUFFER, parts * 4 * sizeof(GLfloat), &initU0[0], GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, axBuf);
	glBufferData(GL_SHADER_STORAGE_BUFFER, parts * sizeof(GLfloat), &axValue[0], GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, u1Bufs);
	glBufferData(GL_SHADER_STORAGE_BUFFER, parts * 4 * sizeof(GLfloat), &initU1[0], GL_DYNAMIC_DRAW);

	// Normal�� ���� ����.
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, normBuf);
	glBufferData(GL_SHADER_STORAGE_BUFFER, parts * 4 * sizeof(GLfloat), NULL, GL_DYNAMIC_COPY);

	// �ﰢ���� �׸��� ���� Index ����.
	glBindBuffer(GL_ARRAY_BUFFER, elBuf);
	glBufferData(GL_ARRAY_BUFFER, el.size() * sizeof(GLuint), &el[0], GL_DYNAMIC_COPY);

	// �ؽ�ó ��ǥ ����.
	glBindBuffer(GL_ARRAY_BUFFER, tcBuf);
	glBufferData(GL_ARRAY_BUFFER, initTc.size() * sizeof(GLfloat), &initTc[0], GL_STATIC_DRAW);

	numElements = GLuint(el.size());

	// VAO ����.
	glGenVertexArrays(1, &waveVao);
	glBindVertexArray(waveVao);

	glBindBuffer(GL_ARRAY_BUFFER, u1Bufs);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, normBuf);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, tcBuf);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elBuf);
	glBindVertexArray(0);

	// �ؽ��� �ε�.
	glGenTextures(1, &textureId);
	glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_2D, textureId);
	My_glTexImage2D_from_file("Resources/water_textile.jpg");

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

void prepareSurface(int N)
{
	surface_vert = (coord*)malloc(sizeof(coord)* (N + 1) * (N + 1));
	normal_vert = (coord*)malloc(sizeof(coord)* (N + 1) * (N + 1));
	int surface_vert_size = sizeof(float*)* N * N * 6 * 3;
	float *surface_vert_arr = (float*)malloc(surface_vert_size);
	surface_n_triangles = N*N * 2;

	for(int i = 0; i <= N; i++)
	{
		for(int j = 0; j <= N; j++)
		{
			surface_vert[i * (N + 1) + j].x = (float)j / (float)(N);
			surface_vert[i * (N + 1) + j].z = (float)i / (float)(N);
			surface_vert[i * (N + 1) + j].y = 0.0f;
		}
	}

	surface_vert_indices = (int *)malloc(sizeof(int)* N * N * 6);
	for(int i = 0, j = 0; j < N*N * 6; i++)
	{
		if((i + 1) % (N + 1) == 0) continue;

		surface_vert_indices[j + 0] = i;
		surface_vert_indices[j + 1] = i + (N + 1);
		surface_vert_indices[j + 2] = i + 1;
		surface_vert_indices[j + 3] = i + (N + 2);
		surface_vert_indices[j + 4] = i + 1;
		surface_vert_indices[j + 5] = i + (N + 1);

		j += 6;
	}

	for(int i = 0, j = 0; i < N*N * 6; i++, j += 3)
	{
		int vIdx = surface_vert_indices[i];

		surface_vert_arr[j] = surface_vert[vIdx].x;
		surface_vert_arr[j + 1] = surface_vert[vIdx].y;
		surface_vert_arr[j + 2] = surface_vert[vIdx].z;
	}

	glGenBuffers(1, &surface_VBO);

	glBindBuffer(GL_ARRAY_BUFFER, surface_VBO);
	glBufferData(GL_ARRAY_BUFFER, surface_vert_size, surface_vert_arr, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenVertexArrays(1, &surface_VAO);
	glBindVertexArray(surface_VAO);
	glBindBuffer(GL_ARRAY_BUFFER, surface_VBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	free(surface_vert_arr);
}

void draw_surface(GLuint hProgramId, glm::mat4 mvpMatrix)
{
	GLfloat object_color[3] = { 1.0f, 0.0f, 0.0f };

	GLuint loc_ModelViewProjectionMatrix_simple = glGetUniformLocation(hProgramId, "ModelViewProjectionMatrix");
	GLuint loc_primitive_color = glGetUniformLocation(hProgramId, "primitive_color");

	glBindVertexArray(surface_VAO);
	// glUniform3fv(loc_primitive_color, 1, object_color);

	glUniformMatrix4fv(loc_ModelViewProjectionMatrix_simple, 1, GL_FALSE, &mvpMatrix[0][0]);
	glDrawArrays(GL_LINES, 0, surface_n_triangles * 3);

	glBindVertexArray(0);
}

void drawSurface(GLuint hProgramId, glm::mat4 mvpMatrix)
{
	GLfloat object_color[3] = { 1.0f, 0.0f, 0.0f };

	GLuint loc_ModelViewProjectionMatrix_simple = glGetUniformLocation(hProgramId, "ModelViewProjectionMatrix");
	GLuint loc_primitive_color = glGetUniformLocation(hProgramId, "primitive_color");
	GLuint loc_texture = glGetUniformLocation(hProgramId, "u_texture");

	glBindVertexArray(waveVao);
	glUniform1i(loc_texture, textureId);
	glUniformMatrix4fv(loc_ModelViewProjectionMatrix_simple, 1, GL_FALSE, &mvpMatrix[0][0]);
	glDrawElements(GL_LINE_STRIP, numElements, GL_UNSIGNED_INT, 0);
	// glDrawElements(GL_TRIANGLE_STRIP, numElements, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}