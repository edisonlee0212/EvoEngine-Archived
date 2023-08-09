layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform EE_GIZMOS_CONSTANTS{
	mat4 model;
	vec4 color;
	float size;
	uint cameraIndex;
};

void main()
{	
	FragColor = color;
}