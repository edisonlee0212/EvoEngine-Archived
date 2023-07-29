layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in vec2 inColor;

void main()
{
    gl_Position = EE_INSTANCES[EE_INSTANCE_INDEX].model * vec4(inPosition, 1.0);
}