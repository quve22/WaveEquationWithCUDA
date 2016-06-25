#version 430

precision highp sampler2D;

in vec4 v_color;
out vec4 o_fragColor;

uniform sampler2D u_texture;

void main(void) 
{
	vec4 val = texture2D(u_texture, vec2(gl_PointCoord.x, gl_PointCoord.y));
	float h = val.y;
	if(h <= 0) h = -h;

	h *= 10.0f;

	o_fragColor = vec4(vec3(h), 1.0);
}