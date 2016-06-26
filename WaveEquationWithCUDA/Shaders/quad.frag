#version 430

precision highp sampler2D;

in vec4 v_color;
out vec4 o_fragColor;

uniform sampler2D u_texture;
uniform sampler2D u_gridTexture;

void main(void) 
{
	vec4 val = texture2D(u_texture, vec2(gl_PointCoord.x, gl_PointCoord.y));
	vec4 gridVal = texture2D(u_gridTexture, vec2(gl_PointCoord.x, gl_PointCoord.y));

	float h = val.y;
	if(h <= 0) h = -h;

	h *= 20.0f;

	o_fragColor = vec4(gridVal.x, gridVal.y, h, 1.0);
}