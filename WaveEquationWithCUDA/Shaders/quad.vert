#version 430

uniform float u_pointSize;

layout(location = 0) in vec4 a_position;

out vec4 v_color;

void main()
{
	v_color = vec4(0.0, 1.0, 0.0, 1.0);

	gl_PointSize = u_pointSize;
	gl_Position = a_position;
}