layout (location = 0) in vec3 inPosition;

layout (location = 0) out vec3 WorldPos;

layout(push_constant) uniform EE_PUSH_CONSTANTS{
	mat4 projection;
	mat4 view;
};

void main()
{
    WorldPos = inPosition;
    gl_Position = projection * view * vec4(inPosition, 1.0);
}