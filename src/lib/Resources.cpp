#include "Resources.hpp"

#include "Cubemap.hpp"
#include "RenderLayer.hpp"
#include "Utilities.hpp"
#include "ProjectManager.hpp"
#include "EditorLayer.hpp"
#include "GeometryStorage.hpp"
#include "Shader.hpp"
#include "TextureStorage.hpp"
using namespace evo_engine;

void Resources::LoadShaders()
{
#pragma region Shaders
#pragma region Shader Includes
	std::string add;
	uint32_t meshWorkGroupInvocations = Graphics::GetInstance().m_meshShaderPropertiesExt.maxPreferredMeshWorkGroupInvocations;
	uint32_t taskWorkGroupInvocations = Graphics::GetInstance().m_meshShaderPropertiesExt.maxPreferredTaskWorkGroupInvocations;

	uint32_t meshSubgroupSize = Graphics::GetInstance().m_vkPhysicalDeviceVulkan11Properties.subgroupSize;
	uint32_t meshSubgroupCount =
		(std::min(std::max(Graphics::Constants::MESHLET_MAX_VERTICES_SIZE, Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE), meshWorkGroupInvocations)
			+ meshSubgroupSize - 1)
		/ meshSubgroupSize;

	uint32_t taskSubgroupSize = Graphics::GetInstance().m_vkPhysicalDeviceVulkan11Properties.subgroupSize;
	uint32_t taskSubgroupCount =
		(taskWorkGroupInvocations + taskSubgroupSize - 1) / taskSubgroupSize;

	taskSubgroupSize = glm::max(taskSubgroupSize, 1u);
	meshSubgroupSize = glm::max(meshSubgroupSize, 1u);
	taskSubgroupCount = glm::max(taskSubgroupCount, 1u);
	meshSubgroupCount = glm::max(meshSubgroupCount, 1u);
	add += "\n#define MAX_DIRECTIONAL_LIGHT_SIZE " + std::to_string(Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE)
		+ "\n#define MAX_KERNEL_AMOUNT " + std::to_string(Graphics::Constants::MAX_KERNEL_AMOUNT)
		+ "\n#define MESHLET_MAX_VERTICES_SIZE " + std::to_string(Graphics::Constants::MESHLET_MAX_VERTICES_SIZE)
		+ "\n#define MESHLET_MAX_TRIANGLES_SIZE " + std::to_string(Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE)
		+ "\n#define MESHLET_MAX_INDICES_SIZE " + std::to_string(Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE * 3)

		+ "\n#define EXT_TASK_SUBGROUP_SIZE " + std::to_string(taskSubgroupSize)
		+ "\n#define EXT_MESH_SUBGROUP_SIZE " + std::to_string(meshSubgroupSize)
		+ "\n#define EXT_TASK_SUBGROUP_COUNT " + std::to_string(taskSubgroupCount)
		+ "\n#define EXT_MESH_SUBGROUP_COUNT " + std::to_string(meshSubgroupCount)

		+ "\n#define EXT_MESHLET_PER_TASK " + std::to_string(taskWorkGroupInvocations)


		+ "\n";

	Graphics::GetInstance().m_shaderBasic = add + FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/Basic.glsl");
	Graphics::GetInstance().m_shaderBasicConstants = add + FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/BasicConstants.glsl");
	Graphics::GetInstance().m_shaderGizmosConstants = add + FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/GizmosConstants.glsl");
	Graphics::GetInstance().m_shaderLighting = FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/Lighting.glsl");
	Graphics::GetInstance().m_shaderSSRConstants = add + FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/SSRConstants.glsl");

#pragma endregion

#pragma region Standard Shader
	{
		auto vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard/Standard.vert");

		auto standardVert = CreateResource<Shader>("STANDARD_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard/StandardSkinned.vert");

		standardVert = CreateResource<Shader>("STANDARD_SKINNED_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard/StandardInstanced.vert");

		standardVert = CreateResource<Shader>("STANDARD_INSTANCED_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard/StandardStrands.vert");

		standardVert = CreateResource<Shader>("STANDARD_STRANDS_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);

		auto tescShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationControl/Standard/StandardStrands.tesc");

		auto standardTesc = CreateResource<Shader>("STANDARD_STRANDS_TESC");
		standardTesc->Set(ShaderType::TessellationControl, tescShaderCode);

		auto teseShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationEvaluation/Standard/StandardStrands.tese");

		auto standardTese = CreateResource<Shader>("STANDARD_STRANDS_TESE");
		standardTese->Set(ShaderType::TessellationEvaluation, teseShaderCode);

		auto geomShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/Standard/StandardStrands.geom");

		auto standardGeom = CreateResource<Shader>("STANDARD_STRANDS_GEOM");
		standardGeom->Set(ShaderType::Geometry, geomShaderCode);

		auto taskShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Task/Standard/Standard.task");
		auto standardTask = CreateResource<Shader>("STANDARD_TASK");
		standardTask->Set(ShaderType::Task, taskShaderCode);

		auto meshShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Mesh/Standard/Standard.mesh");
		auto standardMesh = CreateResource<Shader>("STANDARD_MESH");
		standardMesh->Set(ShaderType::Mesh, meshShaderCode);

		meshShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Mesh/Standard/StandardMeshletColored.mesh");
		standardMesh = CreateResource<Shader>("STANDARD_MESHLET_COLORED_MESH");
		standardMesh->Set(ShaderType::Mesh, meshShaderCode);
	}

#pragma endregion
	auto texPassVertCode = std::string("#version 460\n") +
		FileUtils::LoadFileAsString(
			std::filesystem::path("./DefaultResources") / "Shaders/Vertex/TexturePassThrough.vert");
	auto texPassVert = CreateResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
	texPassVert->Set(ShaderType::Vertex, texPassVertCode);

	auto texPassFragCode = std::string("#version 460\n") +
		FileUtils::LoadFileAsString(
			std::filesystem::path("./DefaultResources") / "Shaders/Fragment/TexturePassThrough.frag");
	auto texPassFrag = CreateResource<Shader>("TEXTURE_PASS_THROUGH_FRAG");
	texPassFrag->Set(ShaderType::Fragment, texPassFragCode);

#pragma region GBuffer
	{
		auto fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + Graphics::GetInstance().m_shaderBasic + "\n" + "\n" + Graphics::GetInstance().m_shaderLighting + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferredLighting.frag");
		auto fragShader = CreateResource<Shader>("STANDARD_DEFERRED_LIGHTING_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);


		fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + Graphics::GetInstance().m_shaderBasic + "\n" + "\n" + Graphics::GetInstance().m_shaderLighting + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferredLightingSceneCamera.frag");
		fragShader = CreateResource<Shader>("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode = std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferred.frag");
		fragShader = CreateResource<Shader>("STANDARD_DEFERRED_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode = std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferredMeshletColored.frag");
		fragShader = CreateResource<Shader>("STANDARD_DEFERRED_MESHLET_COLORED_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion

#pragma region PostProcessing
	{
		auto fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderSSRConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/PostProcessing/SSRReflect.frag");
		auto fragShader = CreateResource<Shader>("SSR_REFLECT_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderSSRConstants + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/PostProcessing/SSRBlur.frag");
		fragShader = CreateResource<Shader>("SSR_BLUR_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderSSRConstants + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/PostProcessing/SSRCombine.frag");
		fragShader = CreateResource<Shader>("SSR_COMBINE_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion
#pragma region Shadow Maps
	{
		auto vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/DirectionalLightShadowMap.vert");

		auto vertShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/DirectionalLightShadowMapSkinned.vert");

		vertShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/DirectionalLightShadowMapInstanced.vert");

		vertShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/DirectionalLightShadowMapStrands.vert");

		vertShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/PointLightShadowMap.vert");

		vertShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/PointLightShadowMapSkinned.vert");

		vertShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/PointLightShadowMapInstanced.vert");

		vertShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/PointLightShadowMapStrands.vert");

		vertShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_STRANDS_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/SpotLightShadowMap.vert");

		vertShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/SpotLightShadowMapSkinned.vert");

		vertShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/SpotLightShadowMapInstanced.vert");

		vertShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/SpotLightShadowMapStrands.vert");

		vertShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_STRANDS_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		auto tescShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationControl/Lighting/ShadowMapStrands.tesc");

		auto tescShader = CreateResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		tescShader->Set(ShaderType::TessellationControl, tescShaderCode);

		auto teseShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationEvaluation/Lighting/ShadowMapStrands.tese");

		auto teseShader = CreateResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		teseShader->Set(ShaderType::TessellationEvaluation, teseShaderCode);

		auto geomShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/Lighting/DirectionalLightShadowMapStrands.geom");

		auto geomShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		geomShader->Set(ShaderType::Geometry, geomShaderCode);

		geomShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/Lighting/PointLightShadowMapStrands.geom");

		geomShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		geomShader->Set(ShaderType::Geometry, geomShaderCode);

		geomShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/Lighting/SpotLightShadowMapStrands.geom");

		geomShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		geomShader->Set(ShaderType::Geometry, geomShaderCode);

		auto taskShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Task/Lighting/DirectionalLightShadowMap.task");

		auto taskShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_TASK");
		taskShader->Set(ShaderType::Task, taskShaderCode);

		taskShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Task/Lighting/PointLightShadowMap.task");

		taskShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_TASK");
		taskShader->Set(ShaderType::Task, taskShaderCode);

		taskShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Task/Lighting/SpotLightShadowMap.task");

		taskShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_TASK");
		taskShader->Set(ShaderType::Task, taskShaderCode);

		auto meshShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Mesh/Lighting/DirectionalLightShadowMap.mesh");

		auto meshShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_MESH");
		meshShader->Set(ShaderType::Mesh, meshShaderCode);

		meshShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Mesh/Lighting/PointLightShadowMap.mesh");

		meshShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_MESH");
		meshShader->Set(ShaderType::Mesh, meshShaderCode);

		meshShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Mesh/Lighting/SpotLightShadowMap.mesh");

		meshShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_MESH");
		meshShader->Set(ShaderType::Mesh, meshShaderCode);


		auto fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderBasicConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Empty.frag");
		auto fragShader = CreateResource<Shader>("EMPTY_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion
#pragma region Environmental
	{
		auto vertShaderCode =
			std::string("#version 460\n") +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/EquirectangularMapToCubemap.vert");
		auto vertShader = CreateResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		auto fragShaderCode =
			std::string("#version 460\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EnvironmentalMapBrdf.frag");
		auto fragShader = CreateResource<Shader>("ENVIRONMENTAL_MAP_BRDF_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 460\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EquirectangularMapToCubemap.frag");
		fragShader = CreateResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 460\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EnvironmentalMapIrradianceConvolution.frag");
		fragShader = CreateResource<Shader>("IRRADIANCE_CONSTRUCT_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 460\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EnvironmentalMapPrefilter.frag");
		fragShader = CreateResource<Shader>("PREFILTER_CONSTRUCT_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion

#pragma region Gizmos
	{
		auto vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/Gizmos.vert");
		auto vertShader = CreateResource<Shader>("GIZMOS_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/GizmosStrands.vert");
		vertShader = CreateResource<Shader>("GIZMOS_STRANDS_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/GizmosInstancedColored.vert");
		vertShader = CreateResource<Shader>("GIZMOS_INSTANCED_COLORED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/GizmosNormalColored.vert");
		vertShader = CreateResource<Shader>("GIZMOS_NORMAL_COLORED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/GizmosStrandsNormalColored.vert");
		vertShader = CreateResource<Shader>("GIZMOS_STRANDS_NORMAL_COLORED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/GizmosVertexColored.vert");
		vertShader = CreateResource<Shader>("GIZMOS_VERTEX_COLORED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		auto tescShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationControl/Gizmos/GizmosStrands.tesc");
		auto tescShader = CreateResource<Shader>("GIZMOS_STRANDS_TESC");
		tescShader->Set(ShaderType::TessellationControl, tescShaderCode);

		tescShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationControl/Gizmos/GizmosStrandsColored.tesc");
		tescShader = CreateResource<Shader>("GIZMOS_STRANDS_COLORED_TESC");
		tescShader->Set(ShaderType::TessellationControl, tescShaderCode);

		auto teseShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationEvaluation/Gizmos/GizmosStrands.tese");
		auto teseShader = CreateResource<Shader>("GIZMOS_STRANDS_TESE");
		teseShader->Set(ShaderType::TessellationEvaluation, teseShaderCode);

		teseShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationEvaluation/Gizmos/GizmosStrandsColored.tese");
		teseShader = CreateResource<Shader>("GIZMOS_STRANDS_COLORED_TESE");
		teseShader->Set(ShaderType::TessellationEvaluation, teseShaderCode);

		auto geomShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/Gizmos/GizmosStrands.geom");
		auto geomShader = CreateResource<Shader>("GIZMOS_STRANDS_GEOM");
		geomShader->Set(ShaderType::Geometry, geomShaderCode);

		geomShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/Gizmos/GizmosStrandsColored.geom");
		geomShader = CreateResource<Shader>("GIZMOS_STRANDS_COLORED_GEOM");
		geomShader->Set(ShaderType::Geometry, geomShaderCode);

		vertShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Gizmos/GizmosStrandsVertexColored.vert");
		vertShader = CreateResource<Shader>("GIZMOS_STRANDS_VERTEX_COLORED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		auto fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Gizmos/Gizmos.frag");
		auto fragShader = CreateResource<Shader>("GIZMOS_FRAG");

		fragShader->Set(ShaderType::Fragment, fragShaderCode);
		fragShaderCode =
			std::string("#version 460\n") + Graphics::GetInstance().m_shaderGizmosConstants + "\n" + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Gizmos/GizmosColored.frag");
		fragShader = CreateResource<Shader>("GIZMOS_COLORED_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion
}

void Resources::LoadPrimitives() const
{
	{
		VertexAttributes attributes{};
		attributes.m_texCoord = true;
		Vertex vertex{};
		std::vector<Vertex> vertices;

		vertex.m_position = { -1, 1, 0 };
		vertex.m_texCoord = { 0, 1 };
		vertices.emplace_back(vertex);

		vertex.m_position = { 1, 1, 0 };
		vertex.m_texCoord = { 1, 1 };
		vertices.emplace_back(vertex);

		vertex.m_position = { -1, -1, 0 };
		vertex.m_texCoord = { 0, 0 };
		vertices.emplace_back(vertex);

		vertex.m_position = { 1, -1, 0 };
		vertex.m_texCoord = { 1, 0 };
		vertices.emplace_back(vertex);

		std::vector<glm::uvec3> triangles = { {0,2,3}, {0,3,1} };
		const auto texPassThrough = CreateResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");
		texPassThrough->SetVertices(attributes, vertices, triangles);
	}
	{
		VertexAttributes attributes{};
		Vertex vertex{};
		std::vector<Vertex> vertices;

		vertex.m_position = { -1, -1, -1 };
		vertices.emplace_back(vertex);//0: 

		vertex.m_position = { 1, 1, -1 };
		vertices.emplace_back(vertex);//1: 

		vertex.m_position = { 1, -1, -1 };
		vertices.emplace_back(vertex);//2: 

		vertex.m_position = { -1, 1, -1 };
		vertices.emplace_back(vertex);//3: 


		vertex.m_position = { -1, -1, 1 };
		vertices.emplace_back(vertex);//4: 

		vertex.m_position = { 1, -1,1 };
		vertices.emplace_back(vertex);//5: 

		vertex.m_position = { 1, 1, 1 };
		vertices.emplace_back(vertex);//6: 

		vertex.m_position = { -1, 1, 1 };
		vertices.emplace_back(vertex);//7: 


		std::vector<glm::uvec3> triangles = {
			{0,1,2}, {1, 0, 3},//OK
			{4,5,6}, {6, 7, 4},//OK
			{7,3,0}, {0, 4, 7},
			{6,2,1}, {2, 6, 5},
			{0,2,5}, {5, 4, 0},
			{3,6,1}, {6, 3, 7},
		};
		const auto renderingCube = CreateResource<Mesh>("PRIMITIVE_RENDERING_CUBE");
		renderingCube->SetVertices(attributes, vertices, triangles);
	}
	{
		const auto quad = CreateResource<Mesh>("PRIMITIVE_QUAD");
		quad->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/quad.evemesh");
	}
	{
		const auto sphere = CreateResource<Mesh>("PRIMITIVE_SPHERE");
		sphere->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/sphere.evemesh");
	}
	{
		const auto cube = CreateResource<Mesh>("PRIMITIVE_CUBE");
		cube->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cube.evemesh");
	}
	{
		const auto cone = CreateResource<Mesh>("PRIMITIVE_CONE");
		cone->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cone.evemesh");
	}
	{
		const auto cylinder = CreateResource<Mesh>("PRIMITIVE_CYLINDER");
		cylinder->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cylinder.evemesh");
	}
	{
		const auto torus = CreateResource<Mesh>("PRIMITIVE_TORUS");
		torus->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/torus.evemesh");
	}
	{
		const auto monkey = CreateResource<Mesh>("PRIMITIVE_MONKEY");
		monkey->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/monkey.evemesh");
	}
	{
		const auto capsule = CreateResource<Mesh>("PRIMITIVE_CAPSULE");
		capsule->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/capsule.evemesh");
	}
}

void Resources::Initialize()
{
	auto& resources = GetInstance();
	resources.m_typedResources.clear();
	resources.m_namedResources.clear();
	resources.m_resources.clear();

	resources.m_currentMaxHandle = Handle(1);

	resources.LoadShaders();


	const auto missingTexture = CreateResource<Texture2D>("TEXTURE_MISSING");
	missingTexture->LoadInternal(std::filesystem::path("./DefaultResources") / "Textures/texture-missing.png");
}

void Resources::InitializeEnvironmentalMap()
{
	auto& resources = GetInstance();
	resources.LoadPrimitives();
	GeometryStorage::DeviceSync();
	const auto defaultEnvironmentalMapTexture = CreateResource<Texture2D>("DEFAULT_ENVIRONMENTAL_MAP_TEXTURE");
	defaultEnvironmentalMapTexture->LoadInternal(std::filesystem::path("./DefaultResources") / "Textures/Cubemaps/GrandCanyon/GCanyon_C_YumaPoint_3k.hdr");

	const auto defaultSkyboxTexture = CreateResource<Texture2D>("DEFAULT_SKYBOX_TEXTURE");
	defaultSkyboxTexture->LoadInternal(std::filesystem::path("./DefaultResources") / "Textures/Cubemaps/GrandCanyon/GCanyon_C_YumaPoint_Env.hdr");

	TextureStorage::DeviceSync();

	const auto defaultSkybox = CreateResource<Cubemap>("DEFAULT_SKYBOX");
	defaultSkybox->Initialize(256);
	defaultSkybox->ConvertFromEquirectangularTexture(defaultSkyboxTexture);

	const auto defaultEnvironmentalMap = CreateResource<EnvironmentalMap>("DEFAULT_ENVIRONMENTAL_MAP");
	defaultEnvironmentalMap->ConstructFromTexture2D(defaultEnvironmentalMapTexture);
}

Handle Resources::GenerateNewHandle()
{
	return m_currentMaxHandle.m_value++;
}

void Resources::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	auto& resources = GetInstance();
	auto& projectManager = ProjectManager::GetInstance();
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("View"))
		{
			ImGui::Checkbox("Assets", &resources.m_showAssets);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
	if (resources.m_showAssets) {
		ImGui::Begin("Assets");
		if (ImGui::BeginTabBar(
			"##Assets", ImGuiTabBarFlags_NoCloseWithMiddleMouseButton))
		{
			if (ImGui::BeginTabItem("Inspection"))
			{
				if (projectManager.m_inspectingAsset)
				{
					auto& asset = projectManager.m_inspectingAsset;
					ImGui::Text("Type:");
					ImGui::SameLine();
					ImGui::Text(asset->GetTypeName().c_str());
					ImGui::Separator();
					ImGui::Text("Name:");
					ImGui::SameLine();
					ImGui::Button(asset->GetTitle().c_str());
					editorLayer->DraggableAsset(asset);
					if (!asset->IsTemporary())
					{
						if (ImGui::Button("Save"))
						{
							asset->Save();
						}
					}
					else
					{
						FileUtils::SaveFile(
							"Allocate path & save",
							asset->GetTypeName(),
							projectManager.m_assetExtensions[asset->GetTypeName()],
							[&](const std::filesystem::path& path) {
								asset->SetPathAndSave(std::filesystem::relative(path, projectManager.m_projectPath));
							},
							true);
					}
					ImGui::SameLine();
					FileUtils::SaveFile(
						"Export...",
						asset->GetTypeName(),
						projectManager.m_assetExtensions[asset->GetTypeName()],
						[&](const std::filesystem::path& path) { asset->Export(path); },
						false);
					ImGui::SameLine();
					FileUtils::OpenFile(
						"Import...",
						asset->GetTypeName(),
						projectManager.m_assetExtensions[asset->GetTypeName()],
						[&](const std::filesystem::path& path) { asset->Import(path); },
						false);

					ImGui::Separator();
					if (asset->OnInspect(editorLayer)) asset->SetUnsaved();
				}
				else
				{
					ImGui::Text("None");
				}
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Shared Assets"))
			{
				if (ImGui::BeginDragDropTarget())
				{
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Asset"))
					{
						IM_ASSERT(payload->DataSize == sizeof(Handle));
						Handle handle = *(Handle*)payload->Data;

						if (const auto asset = ProjectManager::GetAsset(handle))
						{
							AssetRef ref;
							ref.Set(asset);
							resources.m_sharedAssets[asset->GetTypeName()].emplace_back(ref);
						}
					}
					ImGui::EndDragDropTarget();
				}

				for (auto& collection : resources.m_sharedAssets)
				{
					if (ImGui::CollapsingHeader(collection.first.c_str()))
					{
						for (auto it = collection.second.begin(); it != collection.second.end(); ++it)
						{
							auto assetRef = *it;
							const auto ptr = assetRef.Get<IAsset>();
							const std::string tag = "##" + ptr->GetTypeName() + std::to_string(ptr->GetHandle());
							ImGui::Button((ptr->GetTitle() + tag).c_str());

							EditorLayer::Rename(assetRef);
							if (EditorLayer::Remove(assetRef))
							{
								collection.second.erase(it);
								break;
							}
							if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
							{
								projectManager.m_inspectingAsset = ptr;
							}
							EditorLayer::Draggable(assetRef);
						}
					}
				}
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Loaded Assets"))
			{

				for (auto& asset : projectManager.m_loadedAssets)
				{
					if (asset.second->IsTemporary()) continue;
					ImGui::Button(asset.second->GetTitle().c_str());
					editorLayer->DraggableAsset(asset.second);
				}

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Resources"))
			{
				for (auto& collection : resources.m_typedResources)
				{
					if (ImGui::CollapsingHeader(collection.first.c_str()))
					{
						for (auto& i : collection.second)
						{
							ImGui::Button(resources.m_resourceNames[i.second->GetHandle()].c_str());
							editorLayer->DraggableAsset(i.second);
						}
					}
				}
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
		ImGui::End();
	}
}

bool Resources::IsResource(const Handle& handle)
{
	auto& resources = GetInstance();
	return resources.m_resources.find(handle) != resources.m_resources.end();
}

bool Resources::IsResource(const std::shared_ptr<IAsset>& target)
{
	auto& resources = GetInstance();
	return resources.m_resources.find(target->GetHandle()) != resources.m_resources.end();
}

bool Resources::IsResource(const AssetRef& target)
{
	auto& resources = GetInstance();
	return resources.m_resources.find(target.GetAssetHandle()) != resources.m_resources.end();
}