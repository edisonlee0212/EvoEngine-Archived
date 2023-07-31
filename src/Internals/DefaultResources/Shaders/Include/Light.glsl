layout(set = EE_PER_GROUP_SET, binding = 15) uniform samplerCube UE_SKYBOX;
//layout(set = EE_PER_GROUP_SET, binding = 16) uniform samplerCube EE_ENVIRONMENTAL_IRRADIANCE;
//layout(set = EE_PER_GROUP_SET, binding = 17) uniform samplerCube EE_ENVIRONMENTAL_PREFILERED;
layout(set = EE_PER_GROUP_SET, binding = 18) uniform sampler2D EE_ENVIRONMENTAL_BRDFLUT;
layout(set = EE_PER_GROUP_SET, binding = 19) uniform sampler2DArray EE_DIRECTIONAL_LIGHT_SM;
layout(set = EE_PER_GROUP_SET, binding = 20) uniform sampler2DArray EE_POINT_LIGHT_SM;
layout(set = EE_PER_GROUP_SET, binding = 21) uniform sampler2D EE_SPOT_LIGHT_SM;



const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float EE_FUNC_DISTRIBUTION_GGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / max(denom, 0.001); // prevent divide by zero for roughness=0.0 and NdotH=1.0
}
// ----------------------------------------------------------------------------
float EE_FUNC_GEOMETRY_SCHLICK_GGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}
// ----------------------------------------------------------------------------
float EE_FUNC_GEOMETRY_SMITH(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = EE_FUNC_GEOMETRY_SCHLICK_GGX(NdotV, roughness);
	float ggx1 = EE_FUNC_GEOMETRY_SCHLICK_GGX(NdotL, roughness);

	return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 EE_FUNC_FRESNEL_SCHLICK(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// ----------------------------------------------------------------------------
vec3 EE_FUNC_FRESNEL_SCHLICK_ROUGHNESS(float cosTheta, vec3 F0, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 EE_FUNC_CALCULATE_LIGHTS(in bool calculateShadow, vec3 albedo, float specular, float dist, vec3 normal, vec3 viewDir, vec3 fragPos, float metallic, float roughness, vec3 F0);
vec3 EE_FUNC_DIRECTIONAL_LIGHT(vec3 albedo, float specular, int i, vec3 normal, vec3 viewDir, float metallic, float roughness, vec3 F0);
vec3 EE_FUNC_POINT_LIGHT(vec3 albedo, float specular, int i, vec3 normal, vec3 fragPos, vec3 viewDir, float metallic, float roughness, vec3 F0);
vec3 EE_FUNC_SPOT_LIGHT(vec3 albedo, float specular, int i, vec3 normal, vec3 fragPos, vec3 viewDir, float metallic, float roughness, vec3 F0);
float EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(int i, int splitIndex, vec3 fragPos, vec3 normal);
float EE_FUNC_POINT_LIGHT_SHADOW(int i, vec3 fragPos, vec3 normal);
float EE_FUNC_SPOT_LIGHT_SHADOW(int i, vec3 fragPos, vec3 normal);
float EE_LINEARIZE_DEPTH(float ndcDepth);
vec3 EE_DEPTH_TO_CLIP_POS(vec2 texCoords, float ndcDepth);
vec3 EE_DEPTH_TO_WORLD_POS(vec2 texCoords, float ndcDepth);
vec3 EE_DEPTH_TO_VIEW_POS(vec2 texCoords, float ndcDepth);

vec3 EE_FUNC_CALCULATE_ENVIRONMENTAL_LIGHT(vec3 albedo, vec3 normal, vec3 viewDir, float metallic, float roughness, vec3 F0)
{
	// ambient lighting (we now use IBL as the ambient term)
	vec3 F = EE_FUNC_FRESNEL_SCHLICK_ROUGHNESS(max(dot(normal, viewDir), 0.0), F0, roughness);
	vec3 R = reflect(-viewDir, normal);
	vec3 kS = F;
	vec3 kD = 1.0 - kS;
	kD *= 1.0 - metallic;

	//vec3 irradiance = EE_ENVIRONMENTAL_BACKGROUND_COLOR.w == 1.0 ? EE_ENVIRONMENTAL_BACKGROUND_COLOR.xyz * EE_ENVIRONMENTAL_LIGHTING_INTENSITY : pow(texture(EE_ENVIRONMENTAL_IRRADIANCE, normal).rgb, vec3(1.0 / EE_ENVIRONMENTAL_MAP_GAMMA)) * EE_ENVIRONMENTAL_LIGHTING_INTENSITY;
	vec3 irradiance = EE_ENVIRONMENTAL_BACKGROUND_COLOR.xyz * EE_ENVIRONMENTAL_LIGHTING_INTENSITY;
	vec3 diffuse = irradiance * albedo;

	// sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
	const float MAX_REFLECTION_LOD = 4.0;
	//vec3 prefilteredColor = EE_ENVIRONMENTAL_BACKGROUND_COLOR.w == 1.0 ? EE_ENVIRONMENTAL_BACKGROUND_COLOR.xyz * EE_ENVIRONMENTAL_LIGHTING_INTENSITY : pow(textureLod(EE_ENVIRONMENTAL_PREFILERED, R, roughness * MAX_REFLECTION_LOD).rgb, vec3(1.0 / EE_ENVIRONMENTAL_MAP_GAMMA)) * EE_ENVIRONMENTAL_LIGHTING_INTENSITY;
	vec3 prefilteredColor = EE_ENVIRONMENTAL_BACKGROUND_COLOR.xyz * pow(EE_ENVIRONMENTAL_LIGHTING_INTENSITY, EE_GAMMA);
	vec2 brdf = texture(EE_ENVIRONMENTAL_BRDFLUT, vec2(max(dot(normal, viewDir), 0.0), roughness)).rg;
	vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
	vec3 ambient = kD * diffuse + specular;
	return ambient;
}

vec3 EE_FUNC_CALCULATE_LIGHTS(bool calculateShadow, vec3 albedo, float specular, float dist, vec3 normal, vec3 viewDir, vec3 fragPos, float metallic, float roughness, vec3 F0) {
	vec3 result = vec3(0.0, 0.0, 0.0);

	// phase 1: directional lighting
	for (int i = 0; i < EE_DIRECTIONAL_LIGHT_AMOUNT; i++) {
		float shadow = 1.0;
		int lightIndex = EE_CAMERA_INDEX * MAX_DIRECTIONAL_LIGHT_SIZE + i;
		if (calculateShadow && EE_DIRECTIONAL_LIGHTS[lightIndex].diffuse.w == 1.0) {
			int split = 0;
			if (dist < EE_SHADOW_SPLIT_0 - EE_SHADOW_SPLIT_0 * EE_SHADOW_SEAM_FIX_RATIO) {
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 0, fragPos, normal);
			}
			else if (dist < EE_SHADOW_SPLIT_0) {
				//Blend between split 1 & 2
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 0, fragPos, normal);
				float nextLevel = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 1, fragPos, normal);
				shadow = (nextLevel * (dist - (EE_SHADOW_SPLIT_0 - EE_SHADOW_SPLIT_0 * EE_SHADOW_SEAM_FIX_RATIO)) + shadow * (EE_SHADOW_SPLIT_0 - dist)) / (EE_SHADOW_SPLIT_0 * EE_SHADOW_SEAM_FIX_RATIO);
			}
			else if (dist < EE_SHADOW_SPLIT_1 - EE_SHADOW_SPLIT_1 * EE_SHADOW_SEAM_FIX_RATIO) {
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 1, fragPos, normal);
			}
			else if (dist < EE_SHADOW_SPLIT_1) {
				//Blend between split 2 & 3
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 1, fragPos, normal);
				float nextLevel = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 2, fragPos, normal);
				shadow = (nextLevel * (dist - (EE_SHADOW_SPLIT_1 - EE_SHADOW_SPLIT_1 * EE_SHADOW_SEAM_FIX_RATIO)) + shadow * (EE_SHADOW_SPLIT_1 - dist)) / (EE_SHADOW_SPLIT_1 * EE_SHADOW_SEAM_FIX_RATIO);
			}
			else if (dist < EE_SHADOW_SPLIT_2 - EE_SHADOW_SPLIT_2 * EE_SHADOW_SEAM_FIX_RATIO) {
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 2, fragPos, normal);
			}
			else if (dist < EE_SHADOW_SPLIT_2) {
				//Blend between split 3 & 4
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 2, fragPos, normal);
				float nextLevel = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 3, fragPos, normal);
				shadow = (nextLevel * (dist - (EE_SHADOW_SPLIT_2 - EE_SHADOW_SPLIT_2 * EE_SHADOW_SEAM_FIX_RATIO)) + shadow * (EE_SHADOW_SPLIT_2 - dist)) / (EE_SHADOW_SPLIT_2 * EE_SHADOW_SEAM_FIX_RATIO);
			}
			else if (dist < EE_SHADOW_SPLIT_3) {
				shadow = EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(lightIndex, 3, fragPos, normal);
			}
			else {
				shadow = 1.0;
			}
		}
		result += EE_FUNC_DIRECTIONAL_LIGHT(albedo, specular, lightIndex, normal, viewDir, metallic, roughness, F0) * shadow;
	}
	// phase 2: point lights
	for (int i = 0; i < EE_POINT_LIGHT_AMOUNT; i++) {
		float shadow = 1.0;
		if (calculateShadow && EE_POINT_LIGHTS[i].diffuse.w == 1.0) {
			shadow = EE_FUNC_POINT_LIGHT_SHADOW(i, fragPos, normal);
		}
		result += EE_FUNC_POINT_LIGHT(albedo, specular, i, normal, fragPos, viewDir, metallic, roughness, F0) * shadow;
	}
	// phase 3: spot light
	for (int i = 0; i < EE_SPOT_LIGHT_AMOUNT; i++) {
		float shadow = 1.0;
		if (calculateShadow && EE_SPOT_LIGHTS[i].diffuse.w == 1.0) {
			shadow = EE_FUNC_SPOT_LIGHT_SHADOW(i, fragPos, normal);
		}
		result += EE_FUNC_SPOT_LIGHT(albedo, specular, i, normal, fragPos, viewDir, metallic, roughness, F0) * shadow;
	}
	return result;
}

// calculates the color when using a directional light.
vec3 EE_FUNC_DIRECTIONAL_LIGHT(vec3 albedo, float specular, int i, vec3 normal, vec3 viewDir, float metallic, float roughness, vec3 F0)
{
	DirectionalLight light = EE_DIRECTIONAL_LIGHTS[i];
	vec3 lightDir = normalize(-light.direction);
	vec3 H = normalize(viewDir + lightDir);
	vec3 radiance = pow(light.diffuse.xyz, vec3(EE_GAMMA));
	float normalDF = EE_FUNC_DISTRIBUTION_GGX(normal, H, roughness);
	float G = EE_FUNC_GEOMETRY_SMITH(normal, viewDir, lightDir, roughness);
	vec3 F = EE_FUNC_FRESNEL_SCHLICK(clamp(dot(H, viewDir), 0.0, 1.0), F0);
	vec3 nominator = normalDF * G * F;
	float denominator = 4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
	vec3 spec = nominator / max(denominator, 0.001) * specular;
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metallic;
	float NdotL = max(dot(normal, lightDir), 0.0);
	return (kD * albedo / PI + spec) * radiance * NdotL;
}

// calculates the color when using a point light.
vec3 EE_FUNC_POINT_LIGHT(vec3 albedo, float specular, int i, vec3 normal, vec3 fragPos, vec3 viewDir, float metallic, float roughness, vec3 F0)
{
	PointLight light = EE_POINT_LIGHTS[i];
	vec3 lightDir = normalize(light.position - fragPos);
	vec3 H = normalize(viewDir + lightDir);
	float distance = length(light.position - fragPos);
	float attenuation = 1.0 / (light.constantLinearQuadFarPlane.x + light.constantLinearQuadFarPlane.y * distance + light.constantLinearQuadFarPlane.z * (distance * distance));
	vec3 radiance = pow(light.diffuse.xyz, vec3(EE_GAMMA)) * attenuation;
	float normalDF = EE_FUNC_DISTRIBUTION_GGX(normal, H, roughness);
	float G = EE_FUNC_GEOMETRY_SMITH(normal, viewDir, lightDir, roughness);
	vec3 F = EE_FUNC_FRESNEL_SCHLICK(clamp(dot(H, viewDir), 0.0, 1.0), F0);
	vec3 nominator = normalDF * G * F;
	float denominator = 4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
	vec3 spec = nominator / max(denominator, 0.001) * specular;
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metallic;
	float NdotL = max(dot(normal, lightDir), 0.0);
	return (kD * albedo / PI + spec) * radiance * NdotL;

}

// calculates the color when using a spot light.
vec3 EE_FUNC_SPOT_LIGHT(vec3 albedo, float specular, int i, vec3 normal, vec3 fragPos, vec3 viewDir, float metallic, float roughness, vec3 F0)
{
	SpotLight light = EE_SPOT_LIGHTS[i];
	vec3 lightDir = normalize(light.position - fragPos);
	vec3 H = normalize(viewDir + lightDir);
	float distance = length(light.position - fragPos);
	float attenuation = 1.0 / (light.constantLinearQuadFarPlane.x + light.constantLinearQuadFarPlane.y * distance + light.constantLinearQuadFarPlane.z * (distance * distance));
	// spotlight intensity
	float theta = dot(lightDir, normalize(-light.direction));
	float epsilon = light.cutOffOuterCutOffLightSizeBias.x - light.cutOffOuterCutOffLightSizeBias.y;
	float intensity = clamp((theta - light.cutOffOuterCutOffLightSizeBias.y) / epsilon, 0.0, 1.0);

	vec3 radiance = pow(light.diffuse.xyz, vec3(EE_GAMMA)) * attenuation * intensity;
	float normalDF = EE_FUNC_DISTRIBUTION_GGX(normal, H, roughness);
	float G = EE_FUNC_GEOMETRY_SMITH(normal, viewDir, lightDir, roughness);
	vec3 F = EE_FUNC_FRESNEL_SCHLICK(clamp(dot(H, viewDir), 0.0, 1.0), F0);
	vec3 nominator = normalDF * G * F;
	float denominator = 4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
	vec3 spec = nominator / max(denominator, 0.001) * specular;
	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metallic;
	float NdotL = max(dot(normal, lightDir), 0.0);
	return (kD * albedo / PI + spec) * radiance * NdotL;
}

vec2 VogelDiskSample(int sampleIndex, int sampleCount, float phi)
{
	float goldenAngle = 2.4;
	float r = sqrt(sampleIndex + 0.5) / sqrt(sampleCount);
	float theta = goldenAngle * sampleIndex + phi;
	return r * vec2(cos(theta), sin(theta));
}

float InterleavedGradientNoise(vec3 fragCoords) {
	vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	return fract(dot(fragCoords, magic));
}

float EE_FUNC_DIRECTIONAL_LIGHT_SHADOW(int i, int splitIndex, vec3 fragPos, vec3 normal)
{
	DirectionalLight light = EE_DIRECTIONAL_LIGHTS[i];
	vec3 lightDir = light.direction;
	if (dot(lightDir, normal) > -0.02) return 1.0;
	float bias = light.ReservedParameters.z * light.lightFrustumWidth[splitIndex] / light.viewPortXSize;
	float normalOffset = light.ReservedParameters.w * light.lightFrustumWidth[splitIndex] / light.viewPortXSize;

	fragPos = fragPos + normal * normalOffset;
	vec4 fragPosLightSpace = light.lightSpaceMatrix[splitIndex] * vec4(fragPos, 1.0);
	// perform perspective divide
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	//
	if (projCoords.z > 1.0) {
		return 0.0;
	}
	// transform to [0,1] range
	projCoords.x = projCoords.x * 0.5 + 0.5;
	projCoords.y = projCoords.y * 0.5 + 0.5;

	// get depth of current fragment from light's perspective
	projCoords = vec3(projCoords.xy, projCoords.z - bias);
	float shadow = 0.0;
	float lightSize = light.ReservedParameters.x;

	int blockers = 0;
	float avgDistance = 0;

	int sampleAmount = EE_SHADOW_PCSS_BLOCKER_SEARCH_SIZE;
	float sampleWidth = lightSize / light.lightFrustumWidth[splitIndex] / sampleAmount;

	float texScale = float(light.viewPortXSize) / float(textureSize(EE_DIRECTIONAL_LIGHT_SM, 0).x);
	vec2 texBase = vec2(float(light.viewPortXStart) / float(textureSize(EE_DIRECTIONAL_LIGHT_SM, 0).y), float(light.viewPortYStart) / float(textureSize(EE_DIRECTIONAL_LIGHT_SM, 0).y));

	for (int i = -sampleAmount; i <= sampleAmount; i++)
	{
		for (int j = -sampleAmount; j <= sampleAmount; j++) {
			vec2 texCoord = projCoords.xy + vec2(i, j) * sampleWidth;
			float closestDepth = texture(EE_DIRECTIONAL_LIGHT_SM, vec3(texCoord * texScale + texBase, splitIndex)).r;
			int tf = int(closestDepth != 0.0 && projCoords.z > closestDepth);
			avgDistance += closestDepth * tf;
			blockers += tf;
		}
	}
	if (blockers == 0) return 1.0;
	float blockerDistance = avgDistance / blockers;
	float penumbraWidth = (projCoords.z - blockerDistance) / blockerDistance * lightSize;
	float texelSize = penumbraWidth * light.ReservedParameters.x / light.lightFrustumWidth[splitIndex] * light.lightFrustumDistance[splitIndex] / 100.0;

	int shadowCount = 0;
	sampleAmount = EE_SHADOW_SAMPLE_SIZE;
	for (int i = 0; i < sampleAmount; i++)
	{
		vec2 texCoord = projCoords.xy + VogelDiskSample(i, sampleAmount, InterleavedGradientNoise(fragPos * 3141)) * (texelSize + 0.001);
		float closestDepth = texture(EE_DIRECTIONAL_LIGHT_SM, vec3(texCoord * texScale + texBase, splitIndex)).r;
		if (closestDepth == 0.0) continue;
		shadow += projCoords.z < closestDepth ? 1.0 : 0.0;
	}
	shadow /= sampleAmount;
	return shadow;
}

float EE_FUNC_SPOT_LIGHT_SHADOW(int i, vec3 fragPos, vec3 normal) {
	SpotLight light = EE_SPOT_LIGHTS[i];
	float bias = light.cutOffOuterCutOffLightSizeBias.w;
	vec4 fragPosLightSpace = light.lightSpaceMatrix * vec4(fragPos, 1.0);
	fragPosLightSpace.z -= bias;
	vec3 projCoords = (fragPosLightSpace.xyz) / fragPosLightSpace.w;
	
	projCoords.x = projCoords.x * 0.5 + 0.5;
	projCoords.y = projCoords.y * 0.5 + 0.5;

	float texScale = float(light.viewPortXSize) / float(textureSize(EE_SPOT_LIGHT_SM, 0).x);
	vec2 texBase = vec2(float(light.viewPortXStart) / float(textureSize(EE_SPOT_LIGHT_SM, 0).y), float(light.viewPortYStart) / float(textureSize(EE_SPOT_LIGHT_SM, 0).y));

	//Blocker Search
	int sampleAmount = EE_SHADOW_PCSS_BLOCKER_SEARCH_SIZE;
	float lightSize = light.cutOffOuterCutOffLightSizeBias.z * projCoords.z / light.cutOffOuterCutOffLightSizeBias.y;
	float blockers = 0;
	float avgDistance = 0;
	float sampleWidth = lightSize / sampleAmount;
	for (int i = -sampleAmount; i <= sampleAmount; i++)
	{
		for (int j = -sampleAmount; j <= sampleAmount; j++) {
			vec2 texCoord = projCoords.xy + vec2(i, j) * sampleWidth;
			float closestDepth = texture(EE_SPOT_LIGHT_SM, vec2(texCoord * texScale + texBase)).r;
			int tf = int(closestDepth != 0.0 && projCoords.z > closestDepth);
			avgDistance += closestDepth * tf;
			blockers += tf;
		}
	}
	if (blockers == 0) return 1.0;
	float blockerDistance = avgDistance / blockers;
	float penumbraWidth = (projCoords.z - blockerDistance) / blockerDistance * lightSize;
	//End search
	sampleAmount = EE_SHADOW_SAMPLE_SIZE;
	float shadow = 0.0;
	for (int i = 0; i < sampleAmount; i++)
	{
		vec2 texCoord = projCoords.xy + VogelDiskSample(i, sampleAmount, InterleavedGradientNoise(fragPos * 3141)) * (penumbraWidth + 0.001);
		float closestDepth = texture(EE_SPOT_LIGHT_SM, vec2(texCoord * texScale + texBase)).r;
		if (closestDepth == 0.0) continue;
		shadow += projCoords.z < closestDepth ? 1.0 : 0.0;
	}
	shadow /= sampleAmount;
	return shadow;
}

float EE_FUNC_POINT_LIGHT_SHADOW(int i, vec3 fragPos, vec3 normal)
{
	PointLight light = EE_POINT_LIGHTS[i];
	vec3 lightPos = light.position;
	// get vector between fragment position and light position
	vec3 fragToLight = fragPos - lightPos;
	float shadow = 0.0;
	float bias = light.ReservedParameters.x;
	vec3 direction = normalize(fragToLight);
	int slice = 0;
	if (abs(direction.x) >= abs(direction.y) && abs(direction.x) >= abs(direction.z))
	{
		if (direction.x > 0) {
			slice = 0;
		}
		else {
			slice = 1;
		}
	}
	else if (abs(direction.y) >= abs(direction.z)) {
		if (direction.y > 0) {
			slice = 2;
		}
		else {
			slice = 3;
		}
	}
	else {
		if (direction.z > 0) {
			slice = 4;
		}
		else {
			slice = 5;
		}
	}
	vec4 fragPosLightSpace = light.lightSpaceMatrix[slice] * vec4(fragPos, 1.0);
	fragPosLightSpace.z -= bias;
	vec3 projCoords = (fragPosLightSpace.xyz) / fragPosLightSpace.w;
	
	projCoords.x = projCoords.x * 0.5 + 0.5;
	projCoords.y = projCoords.y * 0.5 + 0.5;

	float texScale = float(light.viewPortXSize) / float(textureSize(EE_POINT_LIGHT_SM, 0).x);
	vec2 texBase = vec2(float(light.viewPortXStart) / float(textureSize(EE_POINT_LIGHT_SM, 0).y), float(light.viewPortYStart) / float(textureSize(EE_POINT_LIGHT_SM, 0).y));

	//Blocker Search
	int sampleAmount = EE_SHADOW_PCSS_BLOCKER_SEARCH_SIZE;
	float lightSize = light.ReservedParameters.y * projCoords.z;
	float blockers = 0;
	float avgDistance = 0;
	float sampleWidth = lightSize / sampleAmount;
	for (int i = -sampleAmount; i <= sampleAmount; i++)
	{
		for (int j = -sampleAmount; j <= sampleAmount; j++) {
			vec2 texCoord = projCoords.xy + vec2(i, j) * sampleWidth;
			texCoord.x = clamp(texCoord.x, 1.0 / float(light.viewPortXSize), 1.0 - 1.0 / float(light.viewPortXSize));
			texCoord.y = clamp(texCoord.y, 1.0 / float(light.viewPortXSize), 1.0 - 1.0 / float(light.viewPortXSize));
			float closestDepth = texture(EE_POINT_LIGHT_SM, vec3(texCoord * texScale + texBase, slice)).r;
			int tf = int(closestDepth != 0.0 && projCoords.z > closestDepth);
			avgDistance += closestDepth * tf;
			blockers += tf;
		}
	}

	if (blockers == 0) return 1.0;
	float blockerDistance = avgDistance / blockers;
	float penumbraWidth = (projCoords.z - blockerDistance) / blockerDistance * lightSize;
	//End search
	sampleAmount = EE_SHADOW_SAMPLE_SIZE;
	for (int i = 0; i < sampleAmount; i++)
	{
		vec2 texCoord = projCoords.xy + VogelDiskSample(i, sampleAmount, InterleavedGradientNoise(fragPos * 3141)) * (penumbraWidth + 0.001);
		texCoord.x = clamp(texCoord.x, 1.0 / float(light.viewPortXSize), 1.0 - 1.0 / float(light.viewPortXSize));
		texCoord.y = clamp(texCoord.y, 1.0 / float(light.viewPortXSize), 1.0 - 1.0 / float(light.viewPortXSize));
		float closestDepth = texture(EE_POINT_LIGHT_SM, vec3(texCoord * texScale + texBase, slice)).r;
		if (closestDepth == 0.0) continue;
		shadow += projCoords.z < closestDepth ? 1.0 : 0.0;
	}
	shadow /= sampleAmount;
	return shadow;
}


float EE_PIXEL_DISTANCE(in vec3 worldPosA, in vec3 worldPosB) {
	vec4 coordA = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosA, 1.0);
	vec4 coordB = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosB, 1.0);
	vec2 screenSize = vec2(EE_CAMERA_RESOLUTION_X(), EE_CAMERA_RESOLUTION_Y());
	coordA = coordA / coordA.w;
	coordB = coordB / coordB.w;
	return distance(coordA.xy * screenSize / 2.0, coordB.xy * screenSize / 2.0);
}

int EE_STRANDS_SEGMENT_SUBDIVISION(in vec3 worldPosA, in vec3 worldPosB) {
	vec4 coordA = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosA, 1.0);
	vec4 coordB = EE_CAMERAS[EE_CAMERA_INDEX].EE_CAMERA_PROJECTION_VIEW * vec4(worldPosB, 1.0);
	vec2 screenSize = vec2(EE_CAMERA_RESOLUTION_X(), EE_CAMERA_RESOLUTION_Y());
	coordA = coordA / coordA.w;
	coordB = coordB / coordB.w;
	if (coordA.z < -1.0 && coordB.z < -1.0) return 0;
	float pixelDistance = distance(coordA.xy * screenSize / 2.0, coordB.xy * screenSize / 2.0);
	return max(1, min(EE_STRANDS_SUBDIVISION_MAX_X, int(pixelDistance / EE_STRANDS_SUBDIVISION_X_FACTOR)));
}

void EE_SPLINE_INTERPOLATION(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3, out vec3 result, out vec3 tangent, float t)
{
	float t2 = t * t;
	vec3 a0, a1, a2, a3;

	a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
	a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
	a2 = -0.5 * v0 + 0.5 * v2;
	a3 = v1;

	result = vec3(a0 * t * t2 + a1 * t2 + a2 * t + a3);
	vec3 d1 = vec3(3.0 * a0 * t2 + 2.0 * a1 * t + a2);
	tangent = normalize(d1);
}

void EE_SPLINE_INTERPOLATION(in float v0, in float v1, in float v2, in float v3, out float result, out float tangent, float t)
{
	float t2 = t * t;
	float a0, a1, a2, a3;

	a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
	a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
	a2 = -0.5 * v0 + 0.5 * v2;
	a3 = v1;

	result = a0 * t * t2 + a1 * t2 + a2 * t + a3;
	float d1 = 3.0 * a0 * t2 + 2.0 * a1 * t + a2;
	tangent = d1;
}

void EE_SPLINE_INTERPOLATION(in vec4 v0, in vec4 v1, in vec4 v2, in vec4 v3, out vec4 result, out vec4 tangent, float t)
{
	float t2 = t * t;
	vec4 a0, a1, a2, a3;

	a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
	a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
	a2 = -0.5 * v0 + 0.5 * v2;
	a3 = v1;

	result = vec4(a0 * t * t2 + a1 * t2 + a2 * t + a3);
	vec4 d1 = vec4(3.0 * a0 * t2 + 2.0 * a1 * t + a2);
	tangent = normalize(d1);
}

int EE_STRANDS_RING_SUBDIVISION(in mat4 model, in vec3 worldPos, in vec3 modelPos, in float thickness)
{
	vec3 modelPosB = thickness * vec3(0, 1, 0) + modelPos;
	vec3 endPointWorldPos = (model * vec4(modelPosB, 1.0)).xyz;
	vec3 redirectedWorldPosB = worldPos + EE_CAMERA_UP() * distance(worldPos, endPointWorldPos);
	float subdivision = EE_PIXEL_DISTANCE(worldPos, redirectedWorldPosB);
	return max(3, min(int(subdivision), EE_STRANDS_SUBDIVISION_MAX_Y));
}