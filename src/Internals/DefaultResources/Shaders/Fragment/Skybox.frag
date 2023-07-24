out vec4 FragColor;

in vec3 TexCoord;

void main()
{
	vec3 envColor = EE_CAMERA_CLEAR_COLOR.w == 1.0 ? EE_CAMERA_CLEAR_COLOR.xyz * EE_BACKGROUND_INTENSITY : pow(texture(EE_SKYBOX, TexCoord).rgb, vec3(1.0 / EE_ENVIRONMENTAL_MAP_GAMMA)) * EE_BACKGROUND_INTENSITY;

	envColor = pow(envColor, vec3(1.0 / EE_GAMMA));

	FragColor = vec4(envColor, 1.0);
}