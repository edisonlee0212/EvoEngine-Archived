layout(isolines, equal_spacing) in;

layout (location = 0) in TCS_OUT {
	vec3 FragPos;
	float Thickness;
	vec3 Normal;
	float TexCoord;
} tcs_in[];

layout (location = 0) out TES_OUT {
	vec3 FragPos;
	float Thickness;
	vec3 Normal;
	vec3 Tangent;
	float TexCoord;
} tes_out;

layout(location = 5) in flat uint currentInstanceIndexIn[];
layout(location = 5) out flat uint currentInstanceIndexOut;

void main(){
	currentInstanceIndexOut = currentInstanceIndexIn[0];
	vec3 position, normal, tangent, tempV;
	float thickness, texCoord, tempF;
	EE_SPLINE_INTERPOLATION(tcs_in[0].FragPos, tcs_in[1].FragPos, tcs_in[2].FragPos, tcs_in[3].FragPos, position, tangent, gl_TessCoord.x);
	EE_SPLINE_INTERPOLATION(tcs_in[0].Normal, tcs_in[1].Normal, tcs_in[2].Normal, tcs_in[3].Normal, normal, tempV, gl_TessCoord.x);
	EE_SPLINE_INTERPOLATION(tcs_in[0].TexCoord, tcs_in[1].TexCoord, tcs_in[2].TexCoord, tcs_in[3].TexCoord, texCoord, tempF, gl_TessCoord.x);
	EE_SPLINE_INTERPOLATION(tcs_in[0].Thickness, tcs_in[1].Thickness, tcs_in[2].Thickness, tcs_in[3].Thickness, thickness, tempF, gl_TessCoord.x);

	tes_out.TexCoord = texCoord;
	tes_out.FragPos = position;
	tes_out.Normal = normal;
	tes_out.Thickness = thickness;
	tes_out.Tangent = tangent;
}