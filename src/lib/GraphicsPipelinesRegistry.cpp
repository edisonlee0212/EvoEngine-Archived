#include "Graphics.hpp"
#include "Application.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#include "Shader.hpp"
#include "Mesh.hpp"
#include "ProjectManager.hpp"
#include "EditorLayer.hpp"
#include "PostProcessingStack.hpp"
#include "RenderLayer.hpp"

using namespace EvoEngine;

void Graphics::CreateGraphicsPipelines() const
{
	auto perFrameLayout = GetDescriptorSetLayout("PER_FRAME_LAYOUT");
	auto cameraGBufferLayout = GetDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT");
	auto lightingLayout = GetDescriptorSetLayout("LIGHTING_LAYOUT");

	auto boneMatricesLayout = GetDescriptorSetLayout("BONE_MATRICES_LAYOUT");
	auto instancedDataLayout = GetDescriptorSetLayout("INSTANCED_DATA_LAYOUT");

	if (const auto windowLayer = Application::GetLayer<WindowLayer>()) {
		const auto renderTexturePassThrough = std::make_shared<GraphicsPipeline>();
		renderTexturePassThrough->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		renderTexturePassThrough->m_fragmentShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_FRAG");
		renderTexturePassThrough->m_geometryType = GeometryType::Mesh;
		renderTexturePassThrough->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT"));

		renderTexturePassThrough->m_depthAttachmentFormat = VK_FORMAT_UNDEFINED;
		renderTexturePassThrough->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		renderTexturePassThrough->m_colorAttachmentFormats = { 1, m_swapchain->GetImageFormat() };

		renderTexturePassThrough->PreparePipeline();
		RegisterGraphicsPipeline("RENDER_TEXTURE_PRESENT", renderTexturePassThrough);
	}

	{
		const auto SSRReflect = std::make_shared<GraphicsPipeline>();
		SSRReflect->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		SSRReflect->m_fragmentShader = Resources::GetResource<Shader>("SSR_REFLECT_FRAG");
		SSRReflect->m_geometryType = GeometryType::Mesh;
		SSRReflect->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		SSRReflect->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("SSR_REFLECT_LAYOUT"));

		SSRReflect->m_depthAttachmentFormat = VK_FORMAT_UNDEFINED;
		SSRReflect->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		SSRReflect->m_colorAttachmentFormats = { 2, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = SSRReflect->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(SSRPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		SSRReflect->PreparePipeline();
		RegisterGraphicsPipeline("SSR_REFLECT", SSRReflect);
	}

	{
		const auto SSRBlur = std::make_shared<GraphicsPipeline>();
		SSRBlur->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		SSRBlur->m_fragmentShader = Resources::GetResource<Shader>("SSR_BLUR_FRAG");
		SSRBlur->m_geometryType = GeometryType::Mesh;
		SSRBlur->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("SSR_BLUR_LAYOUT"));

		SSRBlur->m_depthAttachmentFormat = VK_FORMAT_UNDEFINED;
		SSRBlur->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		SSRBlur->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = SSRBlur->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(SSRPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		SSRBlur->PreparePipeline();
		RegisterGraphicsPipeline("SSR_BLUR", SSRBlur);
	}

	{
		const auto SSRCombine = std::make_shared<GraphicsPipeline>();
		SSRCombine->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		SSRCombine->m_fragmentShader = Resources::GetResource<Shader>("SSR_COMBINE_FRAG");
		SSRCombine->m_geometryType = GeometryType::Mesh;
		SSRCombine->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("SSR_COMBINE_LAYOUT"));

		SSRCombine->m_depthAttachmentFormat = VK_FORMAT_UNDEFINED;
		SSRCombine->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		SSRCombine->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = SSRCombine->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(SSRPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		SSRCombine->PreparePipeline();

		RegisterGraphicsPipeline("SSR_COMBINE", SSRCombine);
	}

	{
		const auto standardDeferredPrepass = std::make_shared<GraphicsPipeline>();
		if (Settings::ENABLE_MESH_SHADER) {
			standardDeferredPrepass->m_taskShader = Resources::GetResource<Shader>("STANDARD_TASK");
			standardDeferredPrepass->m_meshShader = Resources::GetResource<Shader>("STANDARD_MESH");
		}
		else
		{
			standardDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_VERT");
		}
		standardDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardDeferredPrepass->m_geometryType = GeometryType::Mesh;
		standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		standardDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_PREPASS", standardDeferredPrepass);
	}
	{
		const auto standardDeferredPrepass = std::make_shared<GraphicsPipeline>();
		if (Settings::ENABLE_MESH_SHADER) {
			standardDeferredPrepass->m_taskShader = Resources::GetResource<Shader>("STANDARD_TASK");
			standardDeferredPrepass->m_meshShader = Resources::GetResource<Shader>("STANDARD_MESHLET_COLORED_MESH");
		}
		else
		{
			standardDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_VERT");
		}
		standardDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_MESHLET_COLORED_FRAG");
		standardDeferredPrepass->m_geometryType = GeometryType::Mesh;
		standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		standardDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_MESHLET_COLORED_PREPASS", standardDeferredPrepass);
	}
	{
		const auto standardSkinnedDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardSkinnedDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_SKINNED_VERT");
		standardSkinnedDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardSkinnedDeferredPrepass->m_geometryType = GeometryType::SkinnedMesh;
		standardSkinnedDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardSkinnedDeferredPrepass->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);

		standardSkinnedDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardSkinnedDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardSkinnedDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardSkinnedDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardSkinnedDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_SKINNED_DEFERRED_PREPASS", standardSkinnedDeferredPrepass);
	}
	{
		const auto standardInstancedDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardInstancedDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_INSTANCED_VERT");
		standardInstancedDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardInstancedDeferredPrepass->m_geometryType = GeometryType::Mesh;
		standardInstancedDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardInstancedDeferredPrepass->m_descriptorSetLayouts.emplace_back(instancedDataLayout);

		standardInstancedDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardInstancedDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardInstancedDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardInstancedDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardInstancedDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_INSTANCED_DEFERRED_PREPASS", standardInstancedDeferredPrepass);
	}
	{
		const auto standardStrandsDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardStrandsDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_STRANDS_VERT");
		standardStrandsDeferredPrepass->m_tessellationControlShader = Resources::GetResource<Shader>("STANDARD_STRANDS_TESC");
		standardStrandsDeferredPrepass->m_tessellationEvaluationShader = Resources::GetResource<Shader>("STANDARD_STRANDS_TESE");
		standardStrandsDeferredPrepass->m_geometryShader = Resources::GetResource<Shader>("STANDARD_STRANDS_GEOM");
		standardStrandsDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardStrandsDeferredPrepass->m_geometryType = GeometryType::Strands;
		standardStrandsDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardStrandsDeferredPrepass->m_descriptorSetLayouts.emplace_back(instancedDataLayout);

		standardStrandsDeferredPrepass->m_tessellationPatchControlPoints = 4;

		standardStrandsDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardStrandsDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardStrandsDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardStrandsDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardStrandsDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_STRANDS_DEFERRED_PREPASS", standardStrandsDeferredPrepass);
	}
	{
		const auto standardDeferredLighting = std::make_shared<GraphicsPipeline>();
		standardDeferredLighting->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		standardDeferredLighting->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_LIGHTING_FRAG");
		standardDeferredLighting->m_geometryType = GeometryType::Mesh;
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(cameraGBufferLayout);
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(lightingLayout);

		standardDeferredLighting->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		standardDeferredLighting->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		standardDeferredLighting->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = standardDeferredLighting->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredLighting->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_LIGHTING", standardDeferredLighting);
	}
	{
		const auto standardDeferredLightingSceneCamera = std::make_shared<GraphicsPipeline>();
		standardDeferredLightingSceneCamera->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		standardDeferredLightingSceneCamera->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA_FRAG");
		standardDeferredLightingSceneCamera->m_geometryType = GeometryType::Mesh;
		standardDeferredLightingSceneCamera->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardDeferredLightingSceneCamera->m_descriptorSetLayouts.emplace_back(cameraGBufferLayout);
		standardDeferredLightingSceneCamera->m_descriptorSetLayouts.emplace_back(lightingLayout);

		standardDeferredLightingSceneCamera->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		standardDeferredLightingSceneCamera->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardDeferredLightingSceneCamera->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = standardDeferredLightingSceneCamera->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardDeferredLightingSceneCamera->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA", standardDeferredLightingSceneCamera);
	}
	{
		const auto directionalLightShadowMap = std::make_shared<GraphicsPipeline>();
		if (Settings::ENABLE_MESH_SHADER) {
			directionalLightShadowMap->m_taskShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_TASK");
			directionalLightShadowMap->m_meshShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_MESH");
		}
		else
		{
			directionalLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_VERT");
		}
		directionalLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMap->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP", directionalLightShadowMap);
	}
	{
		const auto directionalLightShadowMapSkinned = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMapSkinned->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED_VERT");
		directionalLightShadowMapSkinned->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMapSkinned->m_geometryType = GeometryType::SkinnedMesh;
		directionalLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);
		directionalLightShadowMapSkinned->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMapSkinned->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMapSkinned->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMapSkinned->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED", directionalLightShadowMapSkinned);
	}
	{
		const auto directionalLightShadowMapInstanced = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMapInstanced->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		directionalLightShadowMapInstanced->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMapInstanced->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		directionalLightShadowMapInstanced->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMapInstanced->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMapInstanced->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMapInstanced->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_INSTANCED", directionalLightShadowMapInstanced);
	}
	{
		const auto directionalLightShadowMapStrand = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMapStrand->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS_VERT");
		directionalLightShadowMapStrand->m_tessellationControlShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		directionalLightShadowMapStrand->m_tessellationEvaluationShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		directionalLightShadowMapStrand->m_geometryShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		directionalLightShadowMapStrand->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMapStrand->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		directionalLightShadowMapStrand->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMapStrand->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		directionalLightShadowMapStrand->m_tessellationPatchControlPoints = 4;

		auto& pushConstantRange = directionalLightShadowMapStrand->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMapStrand->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS", directionalLightShadowMapStrand);
	}
	{
		const auto pointLightShadowMap = std::make_shared<GraphicsPipeline>();
		if (Settings::ENABLE_MESH_SHADER) {
			pointLightShadowMap->m_taskShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_TASK");
			pointLightShadowMap->m_meshShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_MESH");
		}
		else
		{
			pointLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_VERT");
		}
		pointLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMap->m_geometryType = GeometryType::Mesh;
		pointLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP", pointLightShadowMap);
	}
	{
		const auto pointLightShadowMapSkinned = std::make_shared<GraphicsPipeline>();
		pointLightShadowMapSkinned->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		pointLightShadowMapSkinned->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMapSkinned->m_geometryType = GeometryType::SkinnedMesh;
		pointLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);
		pointLightShadowMapSkinned->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMapSkinned->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMapSkinned->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMapSkinned->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_SKINNED", pointLightShadowMapSkinned);
	}
	{
		const auto pointLightShadowMapInstanced = std::make_shared<GraphicsPipeline>();
		pointLightShadowMapInstanced->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		pointLightShadowMapInstanced->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMapInstanced->m_geometryType = GeometryType::Mesh;
		pointLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		pointLightShadowMapInstanced->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMapInstanced->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMapInstanced->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMapInstanced->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_INSTANCED", pointLightShadowMapInstanced);
	}
	{
		const auto pointLightShadowMapStrand = std::make_shared<GraphicsPipeline>();
		pointLightShadowMapStrand->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_STRANDS_VERT");
		pointLightShadowMapStrand->m_tessellationControlShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		pointLightShadowMapStrand->m_tessellationEvaluationShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		pointLightShadowMapStrand->m_geometryShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		pointLightShadowMapStrand->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMapStrand->m_geometryType = GeometryType::Mesh;
		pointLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		pointLightShadowMapStrand->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMapStrand->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		pointLightShadowMapStrand->m_tessellationPatchControlPoints = 4;

		auto& pushConstantRange = pointLightShadowMapStrand->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMapStrand->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_STRANDS", pointLightShadowMapStrand);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		if (Settings::ENABLE_MESH_SHADER) {
			spotLightShadowMap->m_taskShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_TASK");
			spotLightShadowMap->m_meshShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_MESH");
		}
		else
		{
			spotLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_VERT");
		}
		spotLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMap->m_geometryType = GeometryType::Mesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP", spotLightShadowMap);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		spotLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		spotLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMap->m_geometryType = GeometryType::SkinnedMesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_SKINNED", spotLightShadowMap);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		spotLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		spotLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMap->m_geometryType = GeometryType::Mesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_INSTANCED", spotLightShadowMap);
	}
	{
		const auto spotLightShadowMapStrand = std::make_shared<GraphicsPipeline>();
		spotLightShadowMapStrand->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_STRANDS_VERT");
		spotLightShadowMapStrand->m_tessellationControlShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		spotLightShadowMapStrand->m_tessellationEvaluationShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		spotLightShadowMapStrand->m_geometryShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		spotLightShadowMapStrand->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMapStrand->m_geometryType = GeometryType::Mesh;
		spotLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		spotLightShadowMapStrand->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMapStrand->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		spotLightShadowMapStrand->m_tessellationPatchControlPoints = 4;

		auto& pushConstantRange = spotLightShadowMapStrand->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMapStrand->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_STRANDS", spotLightShadowMapStrand);
	}
	{
		const auto brdfLut = std::make_shared<GraphicsPipeline>();
		brdfLut->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		brdfLut->m_fragmentShader = Resources::GetResource<Shader>("ENVIRONMENTAL_MAP_BRDF_FRAG");
		brdfLut->m_geometryType = GeometryType::Mesh;

		brdfLut->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		brdfLut->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		brdfLut->m_colorAttachmentFormats = { 1, VK_FORMAT_R16G16_SFLOAT };

		brdfLut->PreparePipeline();
		RegisterGraphicsPipeline("ENVIRONMENTAL_MAP_BRDF", brdfLut);
	}
	{
		const auto equirectangularToCubemap = std::make_shared<GraphicsPipeline>();
		equirectangularToCubemap->m_vertexShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		equirectangularToCubemap->m_fragmentShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_FRAG");
		equirectangularToCubemap->m_geometryType = GeometryType::Mesh;

		equirectangularToCubemap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		equirectangularToCubemap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		equirectangularToCubemap->m_colorAttachmentFormats = { 1, Constants::TEXTURE_2D };
		equirectangularToCubemap->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT"));

		auto& pushConstantRange = equirectangularToCubemap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(glm::mat4) + sizeof(float);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		equirectangularToCubemap->PreparePipeline();
		RegisterGraphicsPipeline("EQUIRECTANGULAR_TO_CUBEMAP", equirectangularToCubemap);
	}
	{
		const auto irradianceConstruct = std::make_shared<GraphicsPipeline>();
		irradianceConstruct->m_vertexShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		irradianceConstruct->m_fragmentShader = Resources::GetResource<Shader>("IRRADIANCE_CONSTRUCT_FRAG");
		irradianceConstruct->m_geometryType = GeometryType::Mesh;

		irradianceConstruct->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		irradianceConstruct->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		irradianceConstruct->m_colorAttachmentFormats = { 1, Constants::TEXTURE_2D };
		irradianceConstruct->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT"));

		auto& pushConstantRange = irradianceConstruct->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(glm::mat4) + sizeof(float);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		irradianceConstruct->PreparePipeline();
		RegisterGraphicsPipeline("IRRADIANCE_CONSTRUCT", irradianceConstruct);
	}
	{
		const auto prefilterConstruct = std::make_shared<GraphicsPipeline>();
		prefilterConstruct->m_vertexShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		prefilterConstruct->m_fragmentShader = Resources::GetResource<Shader>("PREFILTER_CONSTRUCT_FRAG");
		prefilterConstruct->m_geometryType = GeometryType::Mesh;

		prefilterConstruct->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		prefilterConstruct->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		prefilterConstruct->m_colorAttachmentFormats = { 1, Constants::TEXTURE_2D };
		prefilterConstruct->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT"));

		auto& pushConstantRange = prefilterConstruct->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(glm::mat4) + sizeof(float);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		prefilterConstruct->PreparePipeline();
		RegisterGraphicsPipeline("PREFILTER_CONSTRUCT", prefilterConstruct);
	}
	{
		const auto gizmos = std::make_shared<GraphicsPipeline>();
		gizmos->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_VERT");
		gizmos->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_FRAG");
		gizmos->m_geometryType = GeometryType::Mesh;

		gizmos->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmos->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmos->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmos->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmos->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmos->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS", gizmos);
	}
	{
		const auto gizmosStrands = std::make_shared<GraphicsPipeline>();
		gizmosStrands->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_VERT");
		gizmosStrands->m_tessellationControlShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_TESC");
		gizmosStrands->m_tessellationEvaluationShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_TESE");
		gizmosStrands->m_geometryShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_GEOM");
		gizmosStrands->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_FRAG");
		gizmosStrands->m_geometryType = GeometryType::Strands;

		gizmosStrands->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosStrands->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosStrands->m_tessellationPatchControlPoints = 4;

		gizmosStrands->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosStrands->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosStrands->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosStrands->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_STRANDS", gizmosStrands);
	}
	{
		const auto gizmosNormalColored = std::make_shared<GraphicsPipeline>();
		gizmosNormalColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_NORMAL_COLORED_VERT");
		gizmosNormalColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosNormalColored->m_geometryType = GeometryType::Mesh;

		gizmosNormalColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosNormalColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosNormalColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosNormalColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		gizmosNormalColored->m_tessellationPatchControlPoints = 4;

		auto& pushConstantRange = gizmosNormalColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosNormalColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_NORMAL_COLORED", gizmosNormalColored);
	}
	{
		const auto gizmosStrandsNormalColored = std::make_shared<GraphicsPipeline>();
		gizmosStrandsNormalColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_NORMAL_COLORED_VERT");
		gizmosStrandsNormalColored->m_tessellationControlShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESC");
		gizmosStrandsNormalColored->m_tessellationEvaluationShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESE");
		gizmosStrandsNormalColored->m_geometryShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_GEOM");
		gizmosStrandsNormalColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosStrandsNormalColored->m_geometryType = GeometryType::Strands;

		gizmosStrandsNormalColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosStrandsNormalColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosStrandsNormalColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosStrandsNormalColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		gizmosStrandsNormalColored->m_tessellationPatchControlPoints = 4;

		auto& pushConstantRange = gizmosStrandsNormalColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosStrandsNormalColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_STRANDS_NORMAL_COLORED", gizmosStrandsNormalColored);
	}
	{
		const auto gizmosVertexColored = std::make_shared<GraphicsPipeline>();
		gizmosVertexColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_VERTEX_COLORED_VERT");
		gizmosVertexColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosVertexColored->m_geometryType = GeometryType::Mesh;

		gizmosVertexColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosVertexColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosVertexColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosVertexColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosVertexColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosVertexColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_VERTEX_COLORED", gizmosVertexColored);
	}
	{
		const auto gizmosStrandsVertexColored = std::make_shared<GraphicsPipeline>();
		gizmosStrandsVertexColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_VERTEX_COLORED_VERT");
		gizmosStrandsVertexColored->m_tessellationControlShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESC");
		gizmosStrandsVertexColored->m_tessellationEvaluationShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESE");
		gizmosStrandsVertexColored->m_geometryShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_GEOM");
		gizmosStrandsVertexColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosStrandsVertexColored->m_geometryType = GeometryType::Strands;

		gizmosStrandsVertexColored->m_tessellationPatchControlPoints = 4;

		gizmosStrandsVertexColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosStrandsVertexColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosStrandsVertexColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosStrandsVertexColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		gizmosStrandsVertexColored->m_tessellationPatchControlPoints = 4;

		auto& pushConstantRange = gizmosStrandsVertexColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosStrandsVertexColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_STRANDS_VERTEX_COLORED", gizmosStrandsVertexColored);
	}
	{
		const auto gizmosInstancedColored = std::make_shared<GraphicsPipeline>();
		gizmosInstancedColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_INSTANCED_COLORED_VERT");
		gizmosInstancedColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosInstancedColored->m_geometryType = GeometryType::Mesh;

		gizmosInstancedColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosInstancedColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosInstancedColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosInstancedColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		gizmosInstancedColored->m_descriptorSetLayouts.emplace_back(instancedDataLayout);

		auto& pushConstantRange = gizmosInstancedColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosInstancedColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_INSTANCED_COLORED", gizmosInstancedColored);
	}
}
