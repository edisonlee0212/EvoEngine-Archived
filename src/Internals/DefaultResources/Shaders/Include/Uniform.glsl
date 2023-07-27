#define EE_PER_FRAME_SET 0
#define EE_PER_PASS_SET 1
#define EE_PER_GROUP_SET 2
#define EE_PER_COMMAND_SET 3

//Lights
struct DirectionalLight {
	vec3 direction;
	vec4 diffuse;
	vec3 specular;
	mat4 lightSpaceMatrix[4];
	vec4 lightFrustumWidth;
	vec4 lightFrustumDistance;
	vec4 ReservedParameters;
	int viewPortXStart;
	int viewPortYStart;
	int viewPortXSize;
	int viewPortYSize;
};

struct PointLight {
	vec3 position;
	vec4 constantLinearQuadFarPlane;
	vec4 diffuse;
	vec3 specular;
	mat4 lightSpaceMatrix[6];
	vec4 ReservedParameters;
	int viewPortXStart;
	int viewPortYStart;
	int viewPortXSize;
	int viewPortYSize;
};

struct SpotLight {
	vec3 position;
	float SpotLightPadding0;

	vec3 direction;
	float SpotLightPadding1;

	mat4 lightSpaceMatrix;
	vec4 cutOffOuterCutOffLightSizeBias;
	vec4 constantLinearQuadFarPlane;
	vec4 diffuse;
	vec3 specular;
	float SpotLightPadding3;
	int viewPortXStart;
	int viewPortYStart;
	int viewPortXSize;
	int viewPortYSize;
};



layout(set = EE_PER_FRAME_SET, binding = 0) uniform EE_RENDERING_SETTINGS_BLOCK
{
	float EE_SHADOW_SPLIT_0;
	float EE_SHADOW_SPLIT_1;
	float EE_SHADOW_SPLIT_2;
	float EE_SHADOW_SPLIT_3;

	int EE_SHADOW_SAMPLE_SIZE;
	int EE_SHADOW_PCSS_BLOCKER_SEARCH_SIZE;
	float EE_SHADOW_SEAM_FIX_RATIO;
	float EE_GAMMA;

	float EE_STRANDS_SUBDIVISION_X_FACTOR;
	float EE_STRANDS_SUBDIVISION_Y_FACTOR;
	int EE_STRANDS_SUBDIVISION_MAX_X;
	int EE_STRANDS_SUBDIVISION_MAX_Y;
};

layout(set = EE_PER_FRAME_SET, binding = 1) uniform EE_ENVIRONMENTAL_BLOCK
{
	vec4 EE_ENVIRONMENTAL_BACKGROUND_COLOR;
	float EE_ENVIRONMENTAL_MAP_GAMMA;
	float EE_ENVIRONMENTAL_LIGHTING_INTENSITY;
	float EE_BACKGROUND_INTENSITY;
	float EE_ENVIRONMENTAL_PADDING2;
};


//Camera
layout(set = EE_PER_PASS_SET, binding = 2) uniform EE_CAMERA
{
	mat4 EE_CAMERA_PROJECTION;
	mat4 EE_CAMERA_VIEW;
	mat4 EE_CAMERA_PROJECTION_VIEW;
	mat4 EE_CAMERA_INVERSE_PROJECTION;
	mat4 EE_CAMERA_INVERSE_VIEW;
	mat4 EE_CAMERA_INVERSE_PROJECTION_VIEW;
	vec4 EE_CAMERA_CLEAR_COLOR;
	vec4 EE_CAMERA_RESERVED1;
	vec4 EE_CAMERA_RESERVED2;
};

layout(set = EE_PER_GROUP_SET, binding = 3) uniform EE_MATERIAL_BLOCK
{
	bool EE_ALBEDO_MAP_ENABLED;
	bool EE_NORMAL_MAP_ENABLED;
	bool EE_METALLIC_MAP_ENABLED;
	bool EE_ROUGHNESS_MAP_ENABLED;
	bool EE_AO_MAP_ENABLED;
	bool EE_CAST_SHADOW;
	bool EE_RECEIVE_SHADOW;
	bool EE_ENABLE_SHADOW;

	vec4 EE_PBR_ALBEDO;
	vec4 EE_PBR_SSSC;
	vec4 EE_PBR_SSSR;
	float EE_PBR_METALLIC;
	float EE_PBR_ROUGHNESS;
	float EE_PBR_AO;
	float EE_PBR_EMISSION;
};




float EE_LINEARIZE_DEPTH(float ndcDepth);
vec3 EE_DEPTH_TO_CLIP_POS(vec2 texCoords, float ndcDepth);
vec3 EE_DEPTH_TO_WORLD_POS(vec2 texCoords, float ndcDepth);
vec3 EE_DEPTH_TO_VIEW_POS(vec2 texCoords, float ndcDepth);



vec3 EE_CAMERA_LEFT() {
	return EE_CAMERA_VIEW[0].xyz;
}

vec3 EE_CAMERA_RIGHT() {
	return -EE_CAMERA_VIEW[0].xyz;
}

vec3 EE_CAMERA_UP() {
	return EE_CAMERA_VIEW[1].xyz;
}

vec3 EE_CAMERA_DOWN() {
	return -EE_CAMERA_VIEW[1].xyz;
}

vec3 EE_CAMERA_BACK() {
	return EE_CAMERA_VIEW[2].xyz;
}

vec3 EE_CAMERA_FRONT() {
	return -EE_CAMERA_VIEW[2].xyz;
}

vec3 EE_CAMERA_POSITION() {
	return EE_CAMERA_INVERSE_VIEW[3].xyz;
}

float EE_CAMERA_NEAR() {
	return EE_CAMERA_RESERVED1.x;
}

float EE_CAMERA_FAR() {
	return EE_CAMERA_RESERVED1.y;
}

float EE_CAMERA_TAN_FOV() {
	return EE_CAMERA_RESERVED1.z;
}

float EE_CAMERA_TAN_HALF_FOV() {
	return EE_CAMERA_RESERVED1.w;
}

float EE_CAMERA_RESOLUTION_X() {
	return EE_CAMERA_RESERVED2.x;
}

float EE_CAMERA_RESOLUTION_Y() {
	return EE_CAMERA_RESERVED2.y;
}

float EE_CAMERA_RESOLUTION_RATIO() {
	return EE_CAMERA_RESERVED2.z;
}

float EE_LINEARIZE_DEPTH(float ndcDepth)
{
	float near = EE_CAMERA_NEAR();
	float far = EE_CAMERA_FAR();
	float z = ndcDepth * 2.0 - 1.0;
	return (2.0 * near * far) / (far + near - z * (far - near));
}

vec3 EE_DEPTH_TO_WORLD_POS(vec2 texCoords, float ndcDepth) {
	vec4 viewPos = vec4(EE_DEPTH_TO_VIEW_POS(texCoords, ndcDepth), 1.0);
	vec4 worldPos = EE_CAMERA_INVERSE_VIEW * viewPos;
	worldPos = worldPos / worldPos.w;
	return worldPos.xyz;
}

vec3 EE_DEPTH_TO_VIEW_POS(vec2 texCoords, float ndcDepth) {
	vec4 clipPos = vec4(EE_DEPTH_TO_CLIP_POS(texCoords, ndcDepth), 1.0);
	vec4 viewPos = EE_CAMERA_INVERSE_PROJECTION * clipPos;
	viewPos = viewPos / viewPos.w;
	return viewPos.xyz;
}

vec3 EE_DEPTH_TO_CLIP_POS(vec2 texCoords, float ndcDepth) {
	vec4 clipPos = vec4(texCoords * 2 - vec2(1), ndcDepth * 2.0 - 1.0, 1.0);
	return clipPos.xyz;
}