
layout(push_constant) uniform EE_GIZMOS_CONSTANTS{
	mat4 EE_MODEL_MATRIX;
	vec4 EE_GIZMO_COLOR;
	float EE_GIZMO_SIZE;
	uint EE_CAMERA_INDEX;
};
