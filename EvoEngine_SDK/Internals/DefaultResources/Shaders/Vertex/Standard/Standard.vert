#extension GL_ARB_shader_draw_parameters : enable
layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in vec4 inColor;

layout(location = 0) out VS_OUT {
	vec3 FragPos;
	vec3 Normal;
	vec3 Tangent;
	vec2 TexCoord;
} vs_out;

layout(location = 5) out flat uint currentInstanceIndex;

void main()
{
	currentInstanceIndex = gl_DrawID + EE_INSTANCE_INDEX;
	vs_out.FragPos = vec3(EE_INSTANCES[currentInstanceIndex].model * vec4(inPosition, 1.0));
	vec3 N = normalize(vec3(EE_INSTANCES[currentInstanceIndex].model * vec4(inNormal,    0.0)));
	vec3 T = normalize(vec3(EE_INSTANCES[currentInstanceIndex].model * vec4(inTangent,   0.0)));
	// re-orthogonalize T with respect to N
	T = normalize(T - dot(T, N) * N);
	vs_out.Normal = N;
	vs_out.Tangent = T;
	vs_out.TexCoord = inTexCoord;
	gl_Position = EE_CAMERAS[EE_CAMERA_INDEX].projection_view * vec4(vs_out.FragPos, 1.0);
}