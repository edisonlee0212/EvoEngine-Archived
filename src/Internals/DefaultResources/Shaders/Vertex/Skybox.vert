layout (location = 0) in vec3 inPosition;

out vec3 TexCoord;

void main()
{
    TexCoord = inPosition;
    vec4 pos = EE_CAMERA_PROJECTION * mat4(mat3(EE_CAMERA_VIEW)) * vec4(inPosition, 1.0);
    gl_Position = pos.xyww;
}  