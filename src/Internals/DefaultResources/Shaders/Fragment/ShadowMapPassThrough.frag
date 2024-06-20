
layout (location = 0) in VS_OUT {
	vec2 TexCoord;
} fs_in;

layout(location = 5) in flat uint currentInstanceIndex;

void main()
{
	uint instanceIndex = currentInstanceIndex;
	MaterialProperties materialProperties = EE_MATERIAL_PROPERTIES[EE_INSTANCES[instanceIndex].materialIndex];
	
	vec2 texCoord = fs_in.TexCoord;
	vec4 albedo = materialProperties.EE_PBR_ALBEDO;
	if (materialProperties.EE_ALBEDO_MAP_INDEX != -1) 
		albedo = texture(EE_TEXTURE_2DS[materialProperties.EE_ALBEDO_MAP_INDEX], texCoord);
	if (albedo.a <= 0.5) discard;
}