layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in vec2 inColor;

void main()
{
    uint currentInstanceIndex = gl_DrawID + EE_INSTANCE_INDEX;
    gl_Position = EE_DIRECTIONAL_LIGHTS[EE_CAMERA_INDEX].lightSpaceMatrix[EE_LIGHT_SPLIT_INDEX] * EE_INSTANCES[currentInstanceIndex].model * EE_INSTANCED_DATA[gl_InstanceIndex].instanceMatrix * vec4(inPosition, 1.0);
}