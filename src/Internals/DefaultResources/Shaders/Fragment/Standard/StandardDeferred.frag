#extension GL_ARB_shader_draw_parameters : enable

precision highp float;
layout (location = 0) in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec3 Tangent;
    vec2 TexCoord;
} fs_in;

layout (location = 0) out vec4 outNormal;
layout (location = 1) out vec4 outAlbedo;
layout (location = 2) out vec4 outMaterial;

layout(location = 5) in flat uint currentInstanceIndex;

void main()
{
    uint instanceIndex = currentInstanceIndex;
    MaterialProperties materialProperties = EE_MATERIAL_PROPERTIES[EE_INSTANCES[instanceIndex].materialIndex];
    
    vec2 texCoord = fs_in.TexCoord;
    vec4 albedo = materialProperties.EE_PBR_ALBEDO;
    if (materialProperties.EE_ALBEDO_MAP_INDEX != -1) 
        albedo = texture(EE_TEXTURE_2DS[materialProperties.EE_ALBEDO_MAP_INDEX], texCoord);
    if (albedo.a < 0.1) discard;

    vec3 B = cross(fs_in.Normal, fs_in.Tangent);
    mat3 TBN = mat3(fs_in.Tangent, B, fs_in.Normal);
    vec3 normal = fs_in.Normal;
   if (materialProperties.EE_NORMAL_MAP_INDEX != -1){
        normal = texture(EE_TEXTURE_2DS[materialProperties.EE_NORMAL_MAP_INDEX], texCoord).rgb;
        normal = normal * 2.0 - 1.0;
        normal = normalize(TBN * normal);
    }

    float roughness = materialProperties.EE_PBR_ROUGHNESS;
    float metallic = materialProperties.EE_PBR_METALLIC;
    float emission = materialProperties.EE_PBR_EMISSION;
    float ao = materialProperties.EE_PBR_AO;

    if (materialProperties.EE_ROUGHNESS_MAP_INDEX != -1) roughness = texture(EE_TEXTURE_2DS[materialProperties.EE_ROUGHNESS_MAP_INDEX], texCoord).r;
    if (materialProperties.EE_METALLIC_MAP_INDEX != -1) metallic = texture(EE_TEXTURE_2DS[materialProperties.EE_METALLIC_MAP_INDEX], texCoord).r;
    if (materialProperties.EE_AO_MAP_INDEX != -1) ao = texture(EE_TEXTURE_2DS[materialProperties.EE_AO_MAP_INDEX], texCoord).r;

    // also store the per-fragment normals into the gbuffer
    outNormal.rgb = normalize((gl_FrontFacing ? 1.0 : -1.0) * normal);
    outNormal.a = instanceIndex + 1;
    outAlbedo = albedo;
    outAlbedo.a = EE_INSTANCES[instanceIndex].infoIndex1;

    outMaterial = vec4(metallic, roughness, emission, ao);
}