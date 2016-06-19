#version 330

uniform mat4 ModelViewProjectionMatrix;
uniform vec3 primitive_color;

layout (location = 0) in vec4 v_position;
layout (location = 1) in vec4 v_normal;

out vec4 color;

const vec2 constantList = vec2(1.0, 0.0);

void main(void) {
	color = vec4(primitive_color, 1.0f);
	gl_Position = ModelViewProjectionMatrix * v_position;
}