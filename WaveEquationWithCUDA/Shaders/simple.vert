#version 430

uniform mat4 ModelViewProjectionMatrix;

layout (location = 0) in vec4 v_position;
layout (location = 1) in vec4 v_normal;
layout (location = 2) in vec2 v_texCoord;

out vec4 Position;
out vec4 Normal;
out vec2 TexCoord;

const vec2 constantList = vec2(1.0, 0.0);

void main(void) {
	TexCoord = v_texCoord;
	gl_Position = ModelViewProjectionMatrix * v_position;
}