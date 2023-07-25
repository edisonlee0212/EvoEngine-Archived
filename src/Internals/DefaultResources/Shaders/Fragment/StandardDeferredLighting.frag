precision highp float;

layout (location = 1) out vec4 FragColor;

layout (location = 0) in VS_OUT {
	vec2 TexCoord;
} fs_in;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inDepth;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormal;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inAlbedo;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inMaterial;

void main()
{
	vec3 normal = 		subpassLoad(inNormal).xyz;
	float ndcDepth = 	subpassLoad(inDepth).x;
	float depth = EE_LINEARIZE_DEPTH(ndcDepth);

	float metallic = 	subpassLoad(inMaterial).x;
	float roughness = 	subpassLoad(inMaterial).y;
	float emission = 	subpassLoad(inMaterial).z;
	float ao = 			subpassLoad(inMaterial).w;

	vec3 albedo = 		subpassLoad(inAlbedo).xyz;

	vec3 fragPos = EE_DEPTH_TO_WORLD_POS(fs_in.TexCoord, ndcDepth);

	vec3 cameraPosition = EE_CAMERA_POSITION();
	vec3 viewDir = normalize(cameraPosition - fragPos);
	bool receiveShadow = true;
	vec3 F0 = vec3(0.04); 
	F0 = mix(F0, albedo, metallic);
	vec3 result = EE_FUNC_CALCULATE_LIGHTS(receiveShadow, albedo, 1.0, depth, normal, viewDir, fragPos, metallic, roughness, F0);
	vec3 ambient = EE_FUNC_CALCULATE_ENVIRONMENTAL_LIGHT(albedo, normal, viewDir, metallic, roughness, F0);
	vec3 color = result + emission * normalize(albedo.xyz) + ambient * ao;
	color = pow(color, vec3(1.0 / EE_GAMMA));
	FragColor = vec4(color, 1.0);
}