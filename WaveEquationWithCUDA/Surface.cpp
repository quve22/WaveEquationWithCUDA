#include "Surface.h"

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

void prepareSurface(int N)
{
	surface_vert = (coord*)malloc(sizeof(coord)* (N + 1) * (N + 1));
	normal_vert = (coord*)malloc(sizeof(coord)* (N + 1) * (N + 1));
	int surface_vert_size = sizeof(float*)* N * N * 6 * 3;
	float *surface_vert_arr = (float*)malloc(surface_vert_size);
	surface_n_triangles = N*N * 2;

	for (int i = 0; i <= N; i++)
	{
		for (int j = 0; j <= N; j++)
		{
			surface_vert[i * (N + 1) + j].x = (float)j / (float)(N);
			surface_vert[i * (N + 1) + j].z = (float)i / (float)(N);
			surface_vert[i * (N + 1) + j].y = 0.0f;
		}
	}

	surface_vert_indices = (int *)malloc(sizeof(int)* N * N * 6);
	for (int i = 0, j = 0; j<N*N * 6; i++)
	{
		if ((i + 1) % (N + 1) == 0) continue;

		surface_vert_indices[j + 0] = i;
		surface_vert_indices[j + 1] = i + (N + 1);
		surface_vert_indices[j + 2] = i + 1;
		surface_vert_indices[j + 3] = i + (N + 2);
		surface_vert_indices[j + 4] = i + 1;
		surface_vert_indices[j + 5] = i + (N + 1);

		j += 6;
	}

	for (int i = 0, j = 0; i<N*N * 6; i++, j += 3)
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
	glUniform3fv(loc_primitive_color, 1, object_color);

	glUniformMatrix4fv(loc_ModelViewProjectionMatrix_simple, 1, GL_FALSE, &mvpMatrix[0][0]);
	glDrawArrays(GL_LINES, 0, surface_n_triangles * 3);

	glBindVertexArray(0);
}