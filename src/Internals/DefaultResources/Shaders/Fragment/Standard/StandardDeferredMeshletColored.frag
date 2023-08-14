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
    uint meshletIndex = currentInstanceIndex;
    vec2 texCoord = fs_in.TexCoord;
    vec4 albedo = vec4(EE_UNIFORM_KERNEL[meshletIndex % MAX_KERNEL_AMOUNT].xyz, 1.0);
    
    vec3 B = cross(fs_in.Normal, fs_in.Tangent);
    mat3 TBN = mat3(fs_in.Tangent, B, fs_in.Normal);
    vec3 normal = fs_in.Normal;
   

    float roughness = 0.0;
    float metallic = 0.0;
    float emission = 0.0;
    float ao = 1.0;

    // also store the per-fragment normals into the gbuffer
    outNormal.rgb = normalize((gl_FrontFacing ? 1.0 : -1.0) * normal);
    outNormal.a = 0;
    outAlbedo = albedo;
    outAlbedo.a = 0;

    outMaterial = vec4(metallic, roughness, emission, ao);
}