#version 430 core

layout (local_size_x = 20, local_size_y = 20) in;

layout(std430, binding=0) buffer Pos {
	vec4 Position[];
};

void main(void)
{
	// Do nothing.
}