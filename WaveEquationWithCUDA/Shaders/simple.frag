#version 330

in vec4 color;

layout (location = 0) out vec4 final_color;

void main(void) {
	final_color = color;
}