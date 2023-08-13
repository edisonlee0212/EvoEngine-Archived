#extension GL_EXT_nonuniform_qualifier : enable
#define EE_PER_FRAME_SET 0
#define EE_PER_PASS_SET 1
#define EE_PER_GROUP_SET 2
#define EE_PER_COMMAND_SET 3

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

	int EE_DIRECTIONAL_LIGHT_AMOUNT;
	int EE_POINT_LIGHT_AMOUNT;
	int EE_SPOT_LIGHT_AMOUNT;
	int EE_ENVIRONMENTAL_BRDFLUT_INDEX;
};

layout(set = EE_PER_FRAME_SET, binding = 1) uniform EE_ENVIRONMENTAL_BLOCK
{
	vec4 EE_ENVIRONMENTAL_BACKGROUND_COLOR;
	float EE_ENVIRONMENTAL_MAP_GAMMA;
	float EE_ENVIRONMENTAL_LIGHTING_INTENSITY;
	float EE_BACKGROUND_INTENSITY;
	float EE_ENVIRONMENTAL_PADDING2;
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

struct MaterialProperties {
	int EE_ALBEDO_MAP_INDEX;
	int EE_NORMAL_MAP_INDEX;
	int EE_METALLIC_MAP_INDEX;
	int EE_ROUGHNESS_MAP_INDEX;
	int EE_AO_MAP_INDEX;
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

layout(set = EE_PER_FRAME_SET, binding = 3) readonly buffer EE_MATERIAL_BLOCK
{
	MaterialProperties EE_MATERIAL_PROPERTIES[];
};

struct Instance {
	mat4 model;
	uint materialIndex;
	uint infoIndex1;
	uint meshletIndexOffset;
	uint meshletSize;
};

layout(set = EE_PER_FRAME_SET, binding = 4) readonly buffer EE_INSTANCE_BLOCK
{
	Instance EE_INSTANCES[];
};

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
layout(set = EE_PER_FRAME_SET, binding = 5) uniform EE_KERNEL_BLOCK
{
	vec4 EE_UNIFORM_KERNEL[MAX_KERNEL_AMOUNT];
	vec4 EE_GAUSS_KERNEL[MAX_KERNEL_AMOUNT];
};

layout(set = EE_PER_FRAME_SET, binding = 6) readonly buffer EE_DIRECTIONAL_LIGHT_BLOCK
{
	DirectionalLight EE_DIRECTIONAL_LIGHTS[];
};

layout(set = EE_PER_FRAME_SET, binding = 7) readonly buffer EE_POINT_LIGHT_BLOCK
{
	PointLight EE_POINT_LIGHTS[];
};

layout(set = EE_PER_FRAME_SET, binding = 8) readonly buffer EE_SPOT_LIGHT_BLOCK
{
	SpotLight EE_SPOT_LIGHTS[];
};

struct Vertex {
	vec4 position;
	vec4 normal;
	vec4 tangent;
	vec4 color;
	vec4 texCoord;
};

Vertex EE_GET_VERTEX(uint chunkIndex, uint vertexIndex);

struct VertexDataChunk {
	Vertex vertices[MESHLET_MAX_VERTICES_SIZE];
};


layout(set = EE_PER_FRAME_SET, binding = 9) readonly buffer EE_VERTICES_BLOCK
{
	VertexDataChunk EE_VERTEX_DATA_CHUNKS[];
};


Vertex EE_GET_VERTEX(uint chunkIndex, uint vertexIndex) {
	return EE_VERTEX_DATA_CHUNKS[chunkIndex].vertices[vertexIndex];
}

struct Meshlet {
	uint primitiveIndices[MESHLET_MAX_INDICES_SIZE];
	uint verticesSize;
	uint triangleSize;
	uint chunkIndex;
};

layout(set = EE_PER_FRAME_SET, binding = 10) readonly buffer EE_MESHLETS_BLOCK
{
	Meshlet EE_MESHLETS[];
};

layout(set = EE_PER_FRAME_SET, binding = 13) uniform sampler2D[] EE_TEXTURE_2DS;
layout(set = EE_PER_FRAME_SET, binding = 14) uniform samplerCube[] EE_CUBEMAPS;

struct InstancedData {
	mat4 instanceMatrix;
	vec4 color;
};

layout(set = EE_PER_PASS_SET, binding = 18) readonly buffer EE_ANIM_BONES_BLOCK
{
	mat4 EE_ANIM_BONES[];
};

layout(set = EE_PER_PASS_SET, binding = 18) readonly buffer EE_INSTANCED_DATA_BLOCK
{
	InstancedData EE_INSTANCED_DATA[];
};

vec3 EE_DEPTH_TO_CLIP_POS(vec2 texCoords, float ndcDepth);
vec3 EE_DEPTH_TO_WORLD_POS(vec2 texCoords, float ndcDepth);
vec3 EE_DEPTH_TO_VIEW_POS(vec2 texCoords, float ndcDepth);

vec3 EE_CAMERA_LEFT() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_VIEW[0].xyz;
}

vec3 EE_CAMERA_RIGHT() {
	return -EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_VIEW[0].xyz;
}

vec3 EE_CAMERA_UP() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_VIEW[1].xyz;
}

vec3 EE_CAMERA_DOWN() {
	return -EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_VIEW[1].xyz;
}

vec3 EE_CAMERA_BACK() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_VIEW[2].xyz;
}

vec3 EE_CAMERA_FRONT() {
	return -EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_VIEW[2].xyz;
}

vec3 EE_CAMERA_POSITION() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_INVERSE_VIEW[3].xyz;
}

float EE_CAMERA_NEAR() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED1.x;
}

float EE_CAMERA_FAR() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED1.y;
}

float EE_CAMERA_TAN_FOV() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED1.z;
}

float EE_CAMERA_TAN_HALF_FOV() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED1.w;
}

float EE_CAMERA_RESOLUTION_X() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED2.x;
}

float EE_CAMERA_RESOLUTION_Y() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED2.y;
}

float EE_CAMERA_RESOLUTION_RATIO() {
	return EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_RESERVED2.z;
}

float EE_LINEARIZE_DEPTH(float ndcDepth)
{
	float near = EE_CAMERA_NEAR();
	float far = EE_CAMERA_FAR();
	return near * far / (far - ndcDepth * (far - near));
}

vec3 EE_DEPTH_TO_WORLD_POS(vec2 texCoords, float ndcDepth) {
	vec4 viewPos = vec4(EE_DEPTH_TO_VIEW_POS(texCoords, ndcDepth), 1.0);
	vec4 worldPos = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_INVERSE_VIEW * viewPos;
	worldPos = worldPos / worldPos.w;
	return worldPos.xyz;
}

vec3 EE_DEPTH_TO_VIEW_POS(vec2 texCoords, float ndcDepth) {
	vec4 clipPos = vec4(EE_DEPTH_TO_CLIP_POS(texCoords, ndcDepth), 1.0);
	vec4 viewPos = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_INVERSE_PROJECTION * clipPos;
	viewPos = viewPos / viewPos.w;
	return viewPos.xyz;
}

vec3 EE_DEPTH_TO_CLIP_POS(vec2 texCoords, float ndcDepth) {
	vec4 clipPos = vec4(texCoords * 2 - vec2(1), ndcDepth, 1.0);
	return clipPos.xyz;
}


float EE_PIXEL_DISTANCE(in vec3 worldPosA, in vec3 worldPosB) {
	vec4 coordA = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosA, 1.0);
	vec4 coordB = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosB, 1.0);
	vec2 screenSize = vec2(EE_CAMERA_RESOLUTION_X(), EE_CAMERA_RESOLUTION_Y());
	coordA = coordA / coordA.w;
	coordB = coordB / coordB.w;
	return distance(coordA.xy * screenSize / 2.0, coordB.xy * screenSize / 2.0);
}

int EE_STRANDS_SEGMENT_SUBDIVISION(in vec3 worldPosA, in vec3 worldPosB) {
	vec4 coordA = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosA, 1.0);
	vec4 coordB = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosB, 1.0);
	vec2 screenSize = vec2(EE_CAMERA_RESOLUTION_X(), EE_CAMERA_RESOLUTION_Y());
	coordA = coordA / coordA.w;
	coordB = coordB / coordB.w;
	if (coordA.z < -1.0 && coordB.z < -1.0) return 0;
	float pixelDistance = distance(coordA.xy * screenSize / 2.0, coordB.xy * screenSize / 2.0);
	return max(1, min(EE_STRANDS_SUBDIVISION_MAX_X, int(pixelDistance / EE_STRANDS_SUBDIVISION_X_FACTOR)));
}

void EE_SPLINE_INTERPOLATION(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3, out vec3 result, out vec3 tangent, float t)
{
	float t2 = t * t;
	vec3 a0, a1, a2, a3;

	a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
	a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
	a2 = -0.5 * v0 + 0.5 * v2;
	a3 = v1;

	result = vec3(a0 * t * t2 + a1 * t2 + a2 * t + a3);
	vec3 d1 = vec3(3.0 * a0 * t2 + 2.0 * a1 * t + a2);
	tangent = normalize(d1);
}

void EE_SPLINE_INTERPOLATION(in float v0, in float v1, in float v2, in float v3, out float result, out float tangent, float t)
{
	float t2 = t * t;
	float a0, a1, a2, a3;

	a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
	a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
	a2 = -0.5 * v0 + 0.5 * v2;
	a3 = v1;

	result = a0 * t * t2 + a1 * t2 + a2 * t + a3;
	float d1 = 3.0 * a0 * t2 + 2.0 * a1 * t + a2;
	tangent = d1;
}

void EE_SPLINE_INTERPOLATION(in vec4 v0, in vec4 v1, in vec4 v2, in vec4 v3, out vec4 result, out vec4 tangent, float t)
{
	float t2 = t * t;
	vec4 a0, a1, a2, a3;

	a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
	a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
	a2 = -0.5 * v0 + 0.5 * v2;
	a3 = v1;

	result = vec4(a0 * t * t2 + a1 * t2 + a2 * t + a3);
	vec4 d1 = vec4(3.0 * a0 * t2 + 2.0 * a1 * t + a2);
	tangent = normalize(d1);
}

int EE_STRANDS_RING_SUBDIVISION(in mat4 model, in vec3 worldPos, in vec3 modelPos, in float thickness)
{
	vec3 modelPosB = thickness * vec3(0, 1, 0) + modelPos;
	vec3 endPointWorldPos = (model * vec4(modelPosB, 1.0)).xyz;
	vec3 redirectedWorldPosB = worldPos + EE_CAMERA_UP() * distance(worldPos, endPointWorldPos);
	float subdivision = EE_PIXEL_DISTANCE(worldPos, redirectedWorldPosB);
	return max(3, min(int(subdivision), EE_STRANDS_SUBDIVISION_MAX_Y));
}