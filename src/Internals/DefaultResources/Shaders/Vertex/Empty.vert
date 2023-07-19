layout (location = 0) in vec3 inPosition;

uniform mat4 model;

void main()
{
	gl_Position = EE_CAMERA_PROJECTION_VIEW * model * vec4(inPosition, 1.0);
}