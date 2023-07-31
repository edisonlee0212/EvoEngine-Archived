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

void main()
{
    MaterialProperties materialProperties = EE_MATERIAL_PROPERTIES[EE_MATERIAL_INDEX];
    
    vec2 texCoord = fs_in.TexCoord;
    vec4 albedo = materialProperties.EE_PBR_ALBEDO;
    if (materialProperties.EE_ALBEDO_MAP_ENABLED) albedo = texture(EE_ALBEDO_MAP, texCoord);
    if (albedo.a < 0.1) discard;

    vec3 B = cross(fs_in.Normal, fs_in.Tangent);
    mat3 TBN = mat3(fs_in.Tangent, B, fs_in.Normal);
    vec3 normal = fs_in.Normal;
   

    float roughness = materialProperties.EE_PBR_ROUGHNESS;
    float metallic = materialProperties.EE_PBR_METALLIC;
    float emission = materialProperties.EE_PBR_EMISSION;
    float ao = materialProperties.EE_PBR_AO;

    if (materialProperties.EE_ROUGHNESS_MAP_ENABLED) roughness = texture(EE_ROUGHNESS_MAP, texCoord).r;
    if (materialProperties.EE_METALLIC_MAP_ENABLED) metallic = texture(EE_METALLIC_MAP, texCoord).r;
    if (materialProperties.EE_AO_MAP_ENABLED) ao = texture(EE_AO_MAP, texCoord).r;

    // also store the per-fragment normals into the gbuffer
    outNormal.rgb = normalize((gl_FrontFacing ? 1.0 : -1.0) * normal);
    outNormal.a = EE_INSTANCE_INDEX;
    outAlbedo = albedo;
    outAlbedo.a = EE_MATERIAL_INDEX;

    outMaterial = vec4(metallic, roughness, emission, ao);
}