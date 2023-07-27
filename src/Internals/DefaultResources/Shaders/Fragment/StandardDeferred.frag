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
    vec2 texCoord = fs_in.TexCoord;
    vec4 albedo = EE_PBR_ALBEDO;
    

    vec3 B = cross(fs_in.Normal, fs_in.Tangent);
    mat3 TBN = mat3(fs_in.Tangent, B, fs_in.Normal);
    vec3 normal = fs_in.Normal;
   

    float roughness = EE_PBR_ROUGHNESS;
    float metallic = EE_PBR_METALLIC;
    float emission = EE_PBR_EMISSION;
    float ao = EE_PBR_AO;

    // also store the per-fragment normals into the gbuffer
    outNormal.rgb = normalize((gl_FrontFacing ? 1.0 : -1.0) * normal);
    outAlbedo = vec4(1.0, 1.0, 1.0, 1.0);
    outMaterial = vec4(metallic, roughness, emission, ao);
}