out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec3 Tangent;
    vec2 TexCoord;
    vec4 Color;
} fs_in;


void main()
{
    vec2 texCoord = fs_in.TexCoord;
    vec4 albedo = EE_PBR_ALBEDO;
    float albedoAlpha = albedo.a;
    if (EE_ALBEDO_MAP_ENABLED) {
        albedo = texture(EE_ALBEDO_MAP, texCoord);
        albedo.a = albedo.a * albedoAlpha;
    }
    if (albedo.a < 0.05)
        discard;

    albedo.xyz = vec3(albedo.xyz * (1.0 - fs_in.Color.w) + fs_in.Color.xyz * fs_in.Color.w);
    
    vec3 B = cross(fs_in.Normal, fs_in.Tangent);
    mat3 TBN = mat3(fs_in.Tangent, B, fs_in.Normal);
    vec3 normal = fs_in.Normal;
    if (EE_NORMAL_MAP_ENABLED){
        normal = texture(EE_NORMAL_MAP, texCoord).rgb;
        normal = normal * 2.0 - 1.0;
        normal = normalize(TBN * normal);
    }

    float roughness = EE_PBR_ROUGHNESS;
    float metallic = EE_PBR_METALLIC;
    float emission = EE_PBR_EMISSION;
    float ao = EE_PBR_AO;

    if (EE_ROUGHNESS_MAP_ENABLED) roughness = texture(EE_ROUGHNESS_MAP, texCoord).r;
    if (EE_METALLIC_MAP_ENABLED) metallic = texture(EE_METALLIC_MAP, texCoord).r;
    if (EE_AO_MAP_ENABLED) ao = texture(EE_AO_MAP, texCoord).r;
    vec3 cameraPosition = EE_CAMERA_POSITION();
    vec3 viewDir = normalize(cameraPosition - fs_in.FragPos);
    float dist = distance(fs_in.FragPos, cameraPosition);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo.rgb, metallic);
    vec3 result = EE_FUNC_CALCULATE_LIGHTS(EE_ENABLE_SHADOW && EE_RECEIVE_SHADOW, albedo.rgb, 1.0, dist, normal, viewDir, fs_in.FragPos, metallic, roughness, F0);
    vec3 ambient = EE_FUNC_CALCULATE_ENVIRONMENTAL_LIGHT(albedo.rgb, normal, viewDir, metallic, roughness, F0);
    vec3 color = result + emission * normalize(albedo.xyz) + ambient * ao;

    color = pow(color, vec3(1.0 / EE_GAMMA));

    FragColor = vec4(color, albedo.a);
}