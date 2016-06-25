#version 430

in vec4 Position;
in vec4 Normal;
in vec2 TexCoord;

uniform sampler2D u_texture;

layout (location = 0) out vec4 final_color;

void main(void) {
	vec4 texColor = texture( u_texture, TexCoord );
	// final_color = vec4(1.0, 0.0, 0.0, 1.0);
	//if(texColor == vec4(0.0, 0.0, 0.0, 1.0))
	//	final_color = vec4(1.0, 0.0, 0.0, 1.0);
	//else
	//final_color = vec4(TexCoord, 0.0, 1.0);
	final_color = texture2D(u_texture, TexCoord);
}