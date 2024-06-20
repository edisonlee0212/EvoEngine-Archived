precision highp float;

layout (location = 0) in VS_OUT {
	vec2 TexCoord;
} fs_in;

layout(set = EE_PER_PASS_SET, binding = 18) uniform sampler2D inDepth;
layout(set = EE_PER_PASS_SET, binding = 19) uniform sampler2D inNormal;
layout(set = EE_PER_PASS_SET, binding = 20) uniform sampler2D inAlbedo;
layout(set = EE_PER_PASS_SET, binding = 21) uniform sampler2D inMaterial;

layout (location = 0) out vec4 FragColor;
void main()
{
	float ndcDepth = 	texture(inDepth, fs_in.TexCoord).x;
	vec3 fragPos = EE_DEPTH_TO_WORLD_POS(fs_in.TexCoord, ndcDepth);
	int infoIndex = int(round(texture(inMaterial, fs_in.TexCoord).z));
	bool instanceSelected = infoIndex == 1;

	if(ndcDepth == 1.0) {
		vec3 cameraPosition = EE_CAMERA_POSITION();
		Camera camera = EE_CAMERAS[EE_CAMERA_INDEX];
		vec3 envColor = EE_SKY_COLOR(fragPos - cameraPosition);
		if(!instanceSelected){
			vec2 texOffset = 1.0 / textureSize(inMaterial, 0); // gets size of single texel
			for(int i = -3; i <= 3; i++){
				for(int j = -3; j <= 3; j++){
					float temp2 = texture(inMaterial, fs_in.TexCoord + vec2(texOffset.x * i, texOffset.y * j)).z;
					int infoIndex = int(round(temp2));
					if(infoIndex == 1){
						FragColor = mix(vec4(1, 0.75, 0.0, 1.0), vec4(envColor, 1.0), 0.1);
						return;
					}
				}
			}
			FragColor = mix(vec4(0.5, 0.5, 0.5, 1.0), vec4(envColor, 1.0), float(EE_LIGHT_SPLIT_INDEX) / 256.0);
		}else{
			FragColor = vec4(envColor, 1.0);
		}
		return;
	}

	float depth = EE_LINEARIZE_DEPTH(ndcDepth);

	vec3 normal = 		texture(inNormal, fs_in.TexCoord).xyz;
	

	int materialIndex = int(round(texture(inMaterial, fs_in.TexCoord).w));
	vec2 materialTexCoord = texture(inMaterial, fs_in.TexCoord).xy;
	MaterialProperties materialProperties = EE_MATERIAL_PROPERTIES[materialIndex];

	float roughness = materialProperties.EE_PBR_ROUGHNESS;
	float metallic = materialProperties.EE_PBR_METALLIC;
	float emission = materialProperties.EE_PBR_EMISSION;
	float ao = materialProperties.EE_PBR_AO;
	vec4 albedo = materialProperties.EE_PBR_ALBEDO;

	if (materialProperties.EE_ROUGHNESS_MAP_INDEX != -1) roughness = texture(EE_TEXTURE_2DS[materialProperties.EE_ROUGHNESS_MAP_INDEX], materialTexCoord).r;
	if (materialProperties.EE_METALLIC_MAP_INDEX != -1) metallic = texture(EE_TEXTURE_2DS[materialProperties.EE_METALLIC_MAP_INDEX], materialTexCoord).r;
	if (materialProperties.EE_AO_MAP_INDEX != -1) ao = texture(EE_TEXTURE_2DS[materialProperties.EE_AO_MAP_INDEX], materialTexCoord).r;
	if (materialProperties.EE_ALBEDO_MAP_INDEX != -1) albedo = texture(EE_TEXTURE_2DS[materialProperties.EE_ALBEDO_MAP_INDEX], materialTexCoord);

	
	vec3 cameraPosition = EE_CAMERA_POSITION();
	vec3 viewDir = normalize(cameraPosition - fragPos);
	bool receiveShadow = true;
	vec3 F0 = vec3(0.04); 
	F0 = mix(F0, albedo.xyz, metallic);
	vec3 result = EE_FUNC_CALCULATE_LIGHTS(receiveShadow, albedo.xyz, 1.0, depth, normal, viewDir, fragPos, metallic, roughness, F0);
	vec3 ambient = EE_FUNC_CALCULATE_ENVIRONMENTAL_LIGHT(albedo.xyz, normal, viewDir, metallic, roughness, F0);
	vec3 color = result + emission * normalize(albedo.xyz) + ambient * ao;
	color = pow(color, vec3(1.0 / EE_GAMMA));

	if(!instanceSelected && EE_INSTANCE_INDEX == 1){
		vec2 texOffset = 1.0 / textureSize(inNormal, 0); // gets size of single texel
		for(int i = -3; i <= 3; i++){
			for(int j = -3; j <= 3; j++){
				float temp2 = texture(inMaterial, fs_in.TexCoord + vec2(texOffset.x * i, texOffset.y * j)).z;
				int infoIndex = int(round(temp2));
				if(infoIndex == 1){
					FragColor = mix(vec4(1, 0.75, 0.0, 1.0), vec4(color, 1.0), 0.1);
					return;
				}
			}
		}
		FragColor = mix(vec4(0.5, 0.5, 0.5, 1.0), vec4(color, 1.0), float(EE_LIGHT_SPLIT_INDEX) / 256.0);
	}else{
		FragColor = vec4(color, 1.0);
	}
	
}