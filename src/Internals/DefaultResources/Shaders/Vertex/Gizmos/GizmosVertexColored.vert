#define EE_PER_FRAME_SET 0
#define EE_PER_PASS_SET 1
#define EE_PER_GROUP_SET 2
#define EE_PER_COMMAND_SET 3

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec2 inTexCoord;
layout (location = 4) in vec4 inColor;

layout(push_constant) uniform EE_GIZMOS_CONSTANTS{
	mat4 model;
	vec4 color;
	float size;
	uint cameraIndex;
};

struct Camera {
	mat4 EE_CAMERA_PROJECTION;
	mat4 EE_CAMERA_VIEW;
	mat4 EE_CAMERA_PROJECTION_VIEW;
	mat4 EE_CAMERA_INVERSE_PROJECTION;
	mat4 EE_CAMERA_INVERSE_VIEW;
	mat4 EE_CAMERA_INVERSE_PROJECTION_VIEW;
	vec4 EE_CAMERA_CLEAR_COLOR;
	vec4 EE_CAMERA_RESERVED1;
	vec4 EE_CAMERA_RESERVED2;
	int EE_SKYBOX_INDEX;
	int EE_ENVIRONMENTAL_IRRADIANCE_INDEX;
	int EE_ENVIRONMENTAL_PREFILERED_INDEX;
	int EE_CAMERA_PADDING1;
};

//Camera
layout(set = EE_PER_FRAME_SET, binding = 2) readonly buffer EE_CAMERA_BLOCK
{
	Camera EE_CAMERAS[];
};

layout(location = 0) out VS_OUT {
	vec4 Color;
} vs_out;

void main()
{
    vs_out.Color = inColor;
	mat4 scaleMatrix = mat4(1.0f);
	gl_Position = EE_CAMERAS[cameraIndex].EE_CAMERA_PROJECTION_VIEW * vec4(vec3(model * scaleMatrix * vec4(inPosition, 1.0)), 1.0);
}