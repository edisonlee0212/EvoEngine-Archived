precision highp float;

layout (location = 0) in VS_OUT {
	vec2 TexCoord;
} fs_in;

layout(set = EE_PER_PASS_SET, binding = 10) uniform sampler2D inDepth;
layout(set = EE_PER_PASS_SET, binding = 11) uniform sampler2D inNormal;
layout(set = EE_PER_PASS_SET, binding = 12) uniform sampler2D inAlbedo;
layout(set = EE_PER_PASS_SET, binding = 13) uniform sampler2D inMaterial;

layout (location = 0) out vec4 FragColor;
void main()
{
	float ndcDepth = 	texture(inDepth, fs_in.TexCoord).x;
	vec3 fragPos = EE_DEPTH_TO_WORLD_POS(fs_in.TexCoord, ndcDepth);
	
	if(ndcDepth == 1.0) {
		vec3 cameraPosition = EE_CAMERA_POSITION();
		Camera camera = EE_CAMERAS[EE_CAMERA_INDEX];
		vec3 envColor = camera.EE_CAMERA_CLEAR_COLOR.w == 1.0 ? 
			camera.EE_CAMERA_CLEAR_COLOR.xyz * EE_BACKGROUND_INTENSITY 
			: pow(texture(UE_SKYBOX, normalize(fragPos - cameraPosition)).rgb, vec3(1.0 / EE_ENVIRONMENTAL_MAP_GAMMA)) * EE_BACKGROUND_INTENSITY;
		
		FragColor = vec4(envColor, 1.0);
		return;
	}

	vec3 normal = 		texture(inNormal, fs_in.TexCoord).xyz;
	float temp =		texture(inAlbedo, fs_in.TexCoord).a;
	int infoIndex = int(round(temp));
	bool instanceSelected = infoIndex == 1;



	float depth = EE_LINEARIZE_DEPTH(ndcDepth);

	float metallic = 	texture(inMaterial, fs_in.TexCoord).x;
	float roughness = 	texture(inMaterial, fs_in.TexCoord).y;
	float emission = 	texture(inMaterial, fs_in.TexCoord).z;
	float ao = 			texture(inMaterial, fs_in.TexCoord).w;

	vec3 albedo = 		texture(inAlbedo, fs_in.TexCoord).xyz;

	

	vec3 cameraPosition = EE_CAMERA_POSITION();
	vec3 viewDir = normalize(cameraPosition - fragPos);
	bool receiveShadow = true;
	vec3 F0 = vec3(0.04); 
	F0 = mix(F0, albedo, metallic);
	vec3 result = EE_FUNC_CALCULATE_LIGHTS(receiveShadow, albedo, 1.0, depth, normal, viewDir, fragPos, metallic, roughness, F0);
	vec3 ambient = EE_FUNC_CALCULATE_ENVIRONMENTAL_LIGHT(albedo, normal, viewDir, metallic, roughness, F0);
	vec3 color = result + emission * normalize(albedo.xyz) + ambient * ao;
	color = pow(color, vec3(1.0 / EE_GAMMA));

	if(!instanceSelected && EE_INSTANCE_INDEX == 1){
		FragColor = mix(vec4(0.5, 0.5, 0.5, 1.0), vec4(color, 1.0), 0.5);
	}else{
		FragColor = vec4(color, 1.0);
	}
	
}