#include "RenderLayer.hpp"
#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#include "ProjectManager.hpp"
//#include "Particles.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "GraphicsPipeline.hpp"
#include "Shader.hpp"
using namespace EvoEngine;

void RenderInstanceCollection::Dispatch(
	const std::function<void(const std::shared_ptr<Material>&)>&
	beginCommandGroupAction,
	const std::function<void(const RenderInstance&)>& commandAction) const
{
	for (auto& renderInstanceGroup : m_renderInstanceGroups) {
		beginCommandGroupAction(renderInstanceGroup.second.m_material);
		for (const auto& renderCommands : renderInstanceGroup.second.m_renderCommands)
		{
			for (const auto& renderCommand : renderCommands.second)
			{
				commandAction(renderCommand);
			}
		}
	}
}

void RenderLayer::OnCreate()
{
	PrepareDescriptorSetLayouts();
	CreateStandardDescriptorBuffers();
	UpdateStandardBindings();

	std::vector<glm::vec4> kernels;
	for (uint32_t i = 0; i < Graphics::StorageSizes::m_maxKernelAmount; i++)
	{
		kernels.emplace_back(glm::ballRand(1.0f), 1.0f);
	}
	for (uint32_t i = 0; i < Graphics::StorageSizes::m_maxKernelAmount; i++)
	{
		kernels.emplace_back(
			glm::gaussRand(0.0f, 1.0f),
			glm::gaussRand(0.0f, 1.0f),
			glm::gaussRand(0.0f, 1.0f),
			glm::gaussRand(0.0f, 1.0f));
	}
	for (int i = 0; i < Graphics::GetMaxFramesInFlight(); i++) {
		memcpy(m_cameraInfoBlockMemory[i], kernels.data(), sizeof(glm::vec4) * kernels.size());
	}
	CreateGraphicsPipelines();

	if (const auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		editorLayer->m_sceneCamera = Serialization::ProduceSerializable<Camera>();
		editorLayer->m_sceneCamera->m_clearColor = glm::vec3(59.0f / 255.0f, 85 / 255.0f, 143 / 255.f);
		editorLayer->m_sceneCamera->m_useClearColor = false;
		editorLayer->m_sceneCamera->OnCreate();
	}

	PrepareEnvironmentalBrdfLut();

	m_shadowMaps = std::make_unique<ShadowMaps>();
	m_shadowMaps->Initialize();
}

void RenderLayer::OnDestroy()
{
	m_renderInfoBlockMemory.clear();
	m_environmentalInfoBlockMemory.clear();
	m_cameraInfoBlockMemory.clear();
	m_materialInfoBlockMemory.clear();
	m_instanceInfoBlockMemory.clear();

	m_renderInfoDescriptorBuffers.clear();
	m_environmentInfoDescriptorBuffers.clear();
	m_cameraInfoDescriptorBuffers.clear();
	m_materialInfoDescriptorBuffers.clear();
	m_objectInfoDescriptorBuffers.clear();
}

void RenderLayer::PreUpdate()
{
	const auto scene = GetScene();
	if (!scene) return;
	m_deferredRenderInstances.m_renderInstanceGroups.clear();
	m_deferredInstancedRenderInstances.m_renderInstanceGroups.clear();
	m_forwardRenderInstances.m_renderInstanceGroups.clear();
	m_forwardInstancedRenderInstances.m_renderInstanceGroups.clear();
	m_transparentRenderInstances.m_renderInstanceGroups.clear();
	m_instancedTransparentRenderInstances.m_renderInstanceGroups.clear();

	m_cameraIndices.clear();
	m_materialIndices.clear();
	m_instanceIndices.clear();
	m_cameraInfoBlocks.clear();
	m_materialInfoBlocks.clear();
	m_instanceInfoBlocks.clear();

	m_directionalLightInfoBlocks.clear();
	m_pointLightInfoBlocks.clear();
	m_spotLightInfoBlocks.clear();

	Bound worldBound;
	std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>> cameras;
	CollectCameras(cameras);
	CollectRenderInstances(worldBound);
	scene->SetBound(worldBound);

	CollectDirectionalLights(cameras);
	CollectPointLights();
	CollectSpotLights();

	//The following data stays consistent during entire frame.
	UploadRenderInfoBlock(m_renderInfoBlock);
	UploadEnvironmentalInfoBlock(m_environmentInfoBlock);

	UploadDirectionalLightInfoBlocks(m_directionalLightInfoBlocks);
	UploadPointLightInfoBlocks(m_pointLightInfoBlocks);
	UploadSpotLightInfoBlocks(m_spotLightInfoBlocks);

	UploadCameraInfoBlocks(m_cameraInfoBlocks);
	UploadMaterialInfoBlocks(m_materialInfoBlocks);
	UploadInstanceInfoBlocks(m_instanceInfoBlocks);

	PreparePointAndSpotLightShadowMap();

	if (const std::shared_ptr<Camera> mainCamera = scene->m_mainCamera.Get<Camera>())
	{
		if (m_allowAutoResize) mainCamera->Resize({ m_mainCameraResolutionX, m_mainCameraResolutionY });
	}
	for (const auto& [cameraGlobalTransform, camera] : cameras)
	{
		camera->m_rendered = false;
		if (camera->m_requireRendering)
		{
			RenderToCamera(camera);
		}
	}
}

void RenderLayer::CreateGraphicsPipelines()
{
	auto perFrameLayout = Graphics::GetDescriptorSetLayout("PER_FRAME_LAYOUT");
	auto pbrTextureLayout = Graphics::GetDescriptorSetLayout("PBR_TEXTURE_LAYOUT");
	auto cameraGBufferLayout = Graphics::GetDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT");
	auto lightingLayout = Graphics::GetDescriptorSetLayout("LIGHTING_LAYOUT");
	{
		const auto standardDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardDeferredPrepass->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_VERT"));
		standardDeferredPrepass->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_DEFERRED_FRAG"));
		standardDeferredPrepass->m_geometryType = GeometryType::Mesh;
		standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(pbrTextureLayout);

		standardDeferredPrepass->m_depthAttachmentFormat = Graphics::ImageFormats::m_gBufferDepth;
		standardDeferredPrepass->m_stencilAttachmentFormat = Graphics::ImageFormats::m_gBufferDepth;
		standardDeferredPrepass->m_colorAttachmentFormats = { 3, Graphics::ImageFormats::m_gBufferColor };

		auto& pushConstantRange = standardDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredPrepass->PreparePipeline();
		Graphics::RegisterGraphicsPipeline("STANDARD_DEFERRED_PREPASS", standardDeferredPrepass);
	}
	{
		const auto standardDeferredLighting = std::make_shared<GraphicsPipeline>();
		standardDeferredLighting->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("TEXTURE_PASS_THROUGH_VERT"));
		standardDeferredLighting->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_DEFERRED_LIGHTING_FRAG"));
		standardDeferredLighting->m_geometryType = GeometryType::Mesh;
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(cameraGBufferLayout);
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(lightingLayout);

		standardDeferredLighting->m_depthAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;
		standardDeferredLighting->m_stencilAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;

		standardDeferredLighting->m_colorAttachmentFormats = { 1, Graphics::ImageFormats::m_renderTextureColor };

		auto& pushConstantRange = standardDeferredLighting->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredLighting->PreparePipeline();
		Graphics::RegisterGraphicsPipeline("STANDARD_DEFERRED_LIGHTING", standardDeferredLighting);
	}
	{
		const auto directionalLightShadowMap = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMap->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("LIGHT_SHADOW_MAP_VERT"));
		directionalLightShadowMap->m_geometryShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("DIRECTIONAL_LIGHT_SHADOW_MAP_GEOM"));
		directionalLightShadowMap->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("EMPTY_FRAG"));
		directionalLightShadowMap->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMap->m_depthAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;
		directionalLightShadowMap->m_stencilAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;

		auto& pushConstantRange = directionalLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMap->PreparePipeline();
		Graphics::RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP", directionalLightShadowMap);
	}
	{
		const auto pointLightShadowMap = std::make_shared<GraphicsPipeline>();
		pointLightShadowMap->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("LIGHT_SHADOW_MAP_VERT"));
		pointLightShadowMap->m_geometryShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("POINT_LIGHT_SHADOW_MAP_GEOM"));
		pointLightShadowMap->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("EMPTY_FRAG"));
		pointLightShadowMap->m_geometryType = GeometryType::Mesh;
		pointLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMap->m_depthAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;
		pointLightShadowMap->m_stencilAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;

		auto& pushConstantRange = pointLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMap->PreparePipeline();
		Graphics::RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP", pointLightShadowMap);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		spotLightShadowMap->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("SPOT_LIGHT_SHADOW_MAP_VERT"));
		spotLightShadowMap->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("EMPTY_FRAG"));
		spotLightShadowMap->m_geometryType = GeometryType::Mesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;
		spotLightShadowMap->m_stencilAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		Graphics::RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP", spotLightShadowMap);
	}
	{
		const auto brdfLut = std::make_shared<GraphicsPipeline>();
		brdfLut->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("TEXTURE_PASS_THROUGH_VERT"));
		brdfLut->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("ENVIRONMENTAL_MAP_BRDF_FRAG"));
		brdfLut->m_geometryType = GeometryType::Mesh;

		brdfLut->m_depthAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;
		brdfLut->m_stencilAttachmentFormat = Graphics::ImageFormats::m_renderTextureDepthStencil;

		brdfLut->m_colorAttachmentFormats = { 1, VK_FORMAT_R16G16_SFLOAT };

		brdfLut->PreparePipeline();
		Graphics::RegisterGraphicsPipeline("ENVIRONMENTAL_MAP_BRDF", brdfLut);
	}
	{
		
	}
}



uint32_t RenderLayer::GetCameraIndex(const Handle& handle)
{
	const auto search = m_cameraIndices.find(handle);
	if (search == m_cameraIndices.end())
	{
		throw std::runtime_error("Unable to find camera!");
	}
	return search->second;
}

uint32_t RenderLayer::GetMaterialIndex(const Handle& handle)
{
	const auto search = m_materialIndices.find(handle);
	if (search == m_materialIndices.end())
	{
		throw std::runtime_error("Unable to find material!");
	}
	return search->second;
}

uint32_t RenderLayer::GetInstanceIndex(const Handle& handle)
{
	const auto search = m_instanceIndices.find(handle);
	if (search == m_instanceIndices.end())
	{
		throw std::runtime_error("Unable to find instance!");
	}
	return search->second;
}

void RenderLayer::UploadCameraInfoBlock(const Handle& handle, const CameraInfoBlock& cameraInfoBlock)
{
	const auto index = GetCameraIndex(handle);
	memcpy(&m_cameraInfoBlockMemory[Graphics::GetCurrentFrameIndex()][index], &cameraInfoBlock, sizeof(CameraInfoBlock));
}

void RenderLayer::UploadMaterialInfoBlock(const Handle& handle, const MaterialInfoBlock& materialInfoBlock)
{
	const auto index = GetMaterialIndex(handle);
	memcpy(&m_materialInfoBlockMemory[Graphics::GetCurrentFrameIndex()][index], &materialInfoBlock, sizeof(MaterialInfoBlock));
}

void RenderLayer::UploadInstanceInfoBlock(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock)
{
	const auto index = GetInstanceIndex(handle);
	memcpy(&m_materialInfoBlockMemory[Graphics::GetCurrentFrameIndex()][index], &instanceInfoBlock, sizeof(InstanceInfoBlock));
}

uint32_t RenderLayer::RegisterCameraIndex(const Handle& handle, const CameraInfoBlock& cameraInfoBlock, bool upload)
{
	const auto search = m_cameraIndices.find(handle);
	if (search == m_cameraIndices.end())
	{
		const uint32_t index = m_cameraInfoBlocks.size();
		m_cameraIndices[handle] = index;
		m_cameraInfoBlocks.emplace_back(cameraInfoBlock);
		return index;
	}
	if (upload)
	{
		UploadCameraInfoBlock(handle, cameraInfoBlock);
	}
	return search->second;
}

uint32_t RenderLayer::RegisterMaterialIndex(const Handle& handle, const MaterialInfoBlock& materialInfoBlock, bool upload)
{
	const auto search = m_materialIndices.find(handle);
	if (search == m_materialIndices.end())
	{
		const uint32_t index = m_materialInfoBlocks.size();
		m_materialIndices[handle] = index;
		m_materialInfoBlocks.emplace_back(materialInfoBlock);
		return index;
	}
	if (upload)
	{
		UploadMaterialInfoBlock(handle, materialInfoBlock);
	}
	return search->second;
}

uint32_t RenderLayer::RegisterInstanceIndex(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock, bool upload)
{
	const auto search = m_instanceIndices.find(handle);
	if (search == m_instanceIndices.end())
	{
		const uint32_t index = m_instanceInfoBlocks.size();
		m_instanceIndices[handle] = index;
		m_instanceInfoBlocks.emplace_back(instanceInfoBlock);
		return index;
	}
	if (upload)
	{
		UploadInstanceInfoBlock(handle, instanceInfoBlock);
	}
	return search->second;
}



void RenderLayer::CollectCameras(std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras)
{
	auto scene = GetScene();
	std::vector<std::pair<std::shared_ptr<Camera>, glm::vec3>> cameraPairs;

	if (auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		if (auto sceneCamera = editorLayer->GetSceneCamera(); sceneCamera || sceneCamera->IsEnabled() || sceneCamera->m_requireRendering)
		{
			cameraPairs.emplace_back(sceneCamera, editorLayer->m_sceneCameraPosition);
			CameraInfoBlock cameraInfoBlock;
			GlobalTransform sceneCameraGT;
			sceneCameraGT.SetValue(editorLayer->m_sceneCameraPosition, editorLayer->m_sceneCameraRotation, glm::vec3(1.0f));
			sceneCamera->UpdateCameraInfoBlock(cameraInfoBlock, sceneCameraGT);
			RegisterCameraIndex(sceneCamera->GetHandle(), cameraInfoBlock);

			cameras.emplace_back(sceneCameraGT, sceneCamera);
		}
	}
	if (const std::vector<Entity>* cameraEntities = scene->UnsafeGetPrivateComponentOwnersList<Camera>())
	{
		for (const auto& i : *cameraEntities)
		{
			if (!scene->IsEntityEnabled(i)) continue;
			assert(scene->HasPrivateComponent<Camera>(i));
			auto camera = scene->GetOrSetPrivateComponent<Camera>(i).lock();
			if (!camera || !camera->IsEnabled() || !camera->m_requireRendering) continue;
			auto cameraGlobalTransform = scene->GetDataComponent<GlobalTransform>(i);
			cameraPairs.emplace_back(camera, cameraGlobalTransform.GetPosition());
			CameraInfoBlock cameraInfoBlock;
			camera->UpdateCameraInfoBlock(cameraInfoBlock, cameraGlobalTransform);
			RegisterCameraIndex(camera->GetHandle(), cameraInfoBlock);

			cameras.emplace_back(cameraGlobalTransform, camera);
		}
	}
}

void RenderLayer::CollectDirectionalLights(const std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras)
{
	const auto scene = GetScene();
	auto sceneBound = scene->GetBound();
	auto& minBound = sceneBound.m_min;
	auto& maxBound = sceneBound.m_max;

	const std::vector<Entity>* directionalLightEntities =
		scene->UnsafeGetPrivateComponentOwnersList<DirectionalLight>();
	m_renderInfoBlock.m_directionalLightSize = 0;
	if (directionalLightEntities && !directionalLightEntities->empty())
	{
		m_directionalLightInfoBlocks.resize(Graphics::StorageSizes::m_maxDirectionalLightSize * cameras.size());
		for (const auto& [cameraGlobalTransform, camera] : cameras)
		{
			size_t directionalLightIndex = 0;
			auto cameraIndex = GetCameraIndex(camera->GetHandle());
			glm::vec3 mainCameraPos = cameraGlobalTransform.GetPosition();
			glm::quat mainCameraRot = cameraGlobalTransform.GetRotation();
			for (const auto& lightEntity : *directionalLightEntities)
			{
				if (!scene->IsEntityEnabled(lightEntity))
					continue;
				const auto dlc = scene->GetOrSetPrivateComponent<DirectionalLight>(lightEntity).lock();
				if (!dlc->IsEnabled())
					continue;
				glm::quat rotation = scene->GetDataComponent<GlobalTransform>(lightEntity).GetRotation();
				glm::vec3 lightDir = glm::normalize(rotation * glm::vec3(0, 0, 1));
				float planeDistance = 0;
				glm::vec3 center;
				const auto blockIndex = cameraIndex * Graphics::StorageSizes::m_maxDirectionalLightSize + directionalLightIndex;
				m_directionalLightInfoBlocks[blockIndex].m_direction = glm::vec4(lightDir, 0.0f);
				m_directionalLightInfoBlocks[blockIndex].m_diffuse =
					glm::vec4(dlc->m_diffuse * dlc->m_diffuseBrightness, dlc->m_castShadow);
				m_directionalLightInfoBlocks[blockIndex].m_specular = glm::vec4(0.0f);
				for (int split = 0; split < 4; split++)
				{
					float splitStart = 0;
					float splitEnd = m_maxShadowDistance;
					if (split != 0)
						splitStart = m_maxShadowDistance * m_shadowCascadeSplit[split - 1];
					if (split != 4 - 1)
						splitEnd = m_maxShadowDistance * m_shadowCascadeSplit[split];
					m_renderInfoBlock.m_splitDistances[split] = splitEnd;
					glm::mat4 lightProjection, lightView;
					float max = 0;
					glm::vec3 lightPos;
					glm::vec3 cornerPoints[8];
					Camera::CalculateFrustumPoints(
						camera, splitStart, splitEnd, mainCameraPos, mainCameraRot, cornerPoints);
					glm::vec3 cameraFrustumCenter =
						(mainCameraRot * glm::vec3(0, 0, -1)) * ((splitEnd - splitStart) / 2.0f + splitStart) +
						mainCameraPos;
					if (m_stableFit)
					{
						// Less detail but no shimmering when rotating the camera.
						// max = glm::distance(cornerPoints[4], cameraFrustumCenter);
						max = splitEnd;
					}
					else
					{
						// More detail but cause shimmering when rotating camera.
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[0],
								Ray::ClosestPointOnLine(cornerPoints[0], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[1],
								Ray::ClosestPointOnLine(cornerPoints[1], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[2],
								Ray::ClosestPointOnLine(cornerPoints[2], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[3],
								Ray::ClosestPointOnLine(cornerPoints[3], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[4],
								Ray::ClosestPointOnLine(cornerPoints[4], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[5],
								Ray::ClosestPointOnLine(cornerPoints[5], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[6],
								Ray::ClosestPointOnLine(cornerPoints[6], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
						max = (glm::max)(
							max,
							glm::distance(
								cornerPoints[7],
								Ray::ClosestPointOnLine(cornerPoints[7], cameraFrustumCenter, cameraFrustumCenter - lightDir)));
					}

					glm::vec3 p0 = Ray::ClosestPointOnLine(
						glm::vec3(maxBound.x, maxBound.y, maxBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);
					glm::vec3 p7 = Ray::ClosestPointOnLine(
						glm::vec3(minBound.x, minBound.y, minBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);

					float d0 = glm::distance(p0, p7);

					glm::vec3 p1 = Ray::ClosestPointOnLine(
						glm::vec3(maxBound.x, maxBound.y, minBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);
					glm::vec3 p6 = Ray::ClosestPointOnLine(
						glm::vec3(minBound.x, minBound.y, maxBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);

					float d1 = glm::distance(p1, p6);

					glm::vec3 p2 = Ray::ClosestPointOnLine(
						glm::vec3(maxBound.x, minBound.y, maxBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);
					glm::vec3 p5 = Ray::ClosestPointOnLine(
						glm::vec3(minBound.x, maxBound.y, minBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);

					float d2 = glm::distance(p2, p5);

					glm::vec3 p3 = Ray::ClosestPointOnLine(
						glm::vec3(maxBound.x, minBound.y, minBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);
					glm::vec3 p4 = Ray::ClosestPointOnLine(
						glm::vec3(minBound.x, maxBound.y, maxBound.z), cameraFrustumCenter, cameraFrustumCenter + lightDir);

					float d3 = glm::distance(p3, p4);

					center = Ray::ClosestPointOnLine(sceneBound.Center(), cameraFrustumCenter, cameraFrustumCenter + lightDir);
					planeDistance = (glm::max)((glm::max)(d0, d1), (glm::max)(d2, d3));
					lightPos = center - lightDir * planeDistance;
					lightView = glm::lookAt(lightPos, lightPos + lightDir, glm::normalize(rotation * glm::vec3(0, 1, 0)));
					lightProjection = glm::ortho(-max, max, -max, max, 0.0f, planeDistance * 2.0f);
					switch (blockIndex)
					{
					case 0:
						m_directionalLightInfoBlocks[blockIndex].m_viewPort = glm::ivec4(
							0, 0, Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2, Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2);
						break;
					case 1:
						m_directionalLightInfoBlocks[blockIndex].m_viewPort = glm::ivec4(
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							0,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2);
						break;
					case 2:
						m_directionalLightInfoBlocks[blockIndex].m_viewPort = glm::ivec4(
							0,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2);
						break;
					case 3:
						m_directionalLightInfoBlocks[blockIndex].m_viewPort = glm::ivec4(
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2,
							Graphics::StorageSizes::m_directionalLightShadowMapResolution / 2);
						break;
					}

#pragma region Fix Shimmering due to the movement of the camera

					glm::mat4 shadowMatrix = lightProjection * lightView;
					glm::vec4 shadowOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
					shadowOrigin = shadowMatrix * shadowOrigin;
					GLfloat storedW = shadowOrigin.w;
					shadowOrigin = shadowOrigin * (float)m_directionalLightInfoBlocks[blockIndex].m_viewPort.z / 2.0f;
					glm::vec4 roundedOrigin = glm::round(shadowOrigin);
					glm::vec4 roundOffset = roundedOrigin - shadowOrigin;
					roundOffset = roundOffset * 2.0f / (float)m_directionalLightInfoBlocks[blockIndex].m_viewPort.z;
					roundOffset.z = 0.0f;
					roundOffset.w = 0.0f;
					glm::mat4 shadowProj = lightProjection;
					shadowProj[3] += roundOffset;
					lightProjection = shadowProj;
#pragma endregion
					m_directionalLightInfoBlocks[blockIndex].m_lightSpaceMatrix[split] = lightProjection * lightView;
					m_directionalLightInfoBlocks[blockIndex].m_lightFrustumWidth[split] = max;
					m_directionalLightInfoBlocks[blockIndex].m_lightFrustumDistance[split] = planeDistance;
					if (split == 4 - 1)
						m_directionalLightInfoBlocks[blockIndex].m_reservedParameters =
						glm::vec4(dlc->m_lightSize, 0, dlc->m_bias, dlc->m_normalOffset);
				}

				directionalLightIndex++;
			}

			m_renderInfoBlock.m_directionalLightSize = directionalLightIndex;
		}
	}
}

void RenderLayer::CollectPointLights()
{
	const auto scene = GetScene();
	const std::vector<Entity>* pointLightEntities =
		scene->UnsafeGetPrivateComponentOwnersList<PointLight>();
	m_renderInfoBlock.m_pointLightSize = 0;
	if (pointLightEntities && !pointLightEntities->empty())
	{
		m_pointLightInfoBlocks.resize(pointLightEntities->size());
		for (int i = 0; i < pointLightEntities->size(); i++)
		{
			Entity lightEntity = pointLightEntities->at(i);
			if (!scene->IsEntityEnabled(lightEntity))
				continue;
			const auto plc = scene->GetOrSetPrivateComponent<PointLight>(lightEntity).lock();
			if (!plc->IsEnabled())
				continue;
			glm::vec3 position = scene->GetDataComponent<GlobalTransform>(lightEntity).m_value[3];
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_position = glm::vec4(position, 0);
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_constantLinearQuadFarPlane.x = plc->m_constant;
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_constantLinearQuadFarPlane.y = plc->m_linear;
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_constantLinearQuadFarPlane.z = plc->m_quadratic;
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_diffuse =
				glm::vec4(plc->m_diffuse * plc->m_diffuseBrightness, plc->m_castShadow);
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_specular = glm::vec4(0);
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_constantLinearQuadFarPlane.w = plc->GetFarPlane();

			glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), 1.0f,
				1.0f,
				m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_constantLinearQuadFarPlane.w);
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_lightSpaceMatrix[0] =
				shadowProj *
				glm::lookAt(position, position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_lightSpaceMatrix[1] =
				shadowProj *
				glm::lookAt(position, position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_lightSpaceMatrix[2] =
				shadowProj * glm::lookAt(position, position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_lightSpaceMatrix[3] =
				shadowProj *
				glm::lookAt(position, position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_lightSpaceMatrix[4] =
				shadowProj *
				glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_lightSpaceMatrix[5] =
				shadowProj *
				glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
			m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_reservedParameters = glm::vec4(plc->m_bias, plc->m_lightSize, 0, 0);

			switch (m_renderInfoBlock.m_pointLightSize)
			{
			case 0:
				m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_viewPort =
					glm::ivec4(0, 0, Graphics::StorageSizes::m_pointLightShadowMapResolution / 2, Graphics::StorageSizes::m_pointLightShadowMapResolution / 2);
				break;
			case 1:
				m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_viewPort = glm::ivec4(
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					0,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2);
				break;
			case 2:
				m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_viewPort = glm::ivec4(
					0,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2);
				break;
			case 3:
				m_pointLightInfoBlocks[m_renderInfoBlock.m_pointLightSize].m_viewPort = glm::ivec4(
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_pointLightShadowMapResolution / 2);
				break;
			}
			m_renderInfoBlock.m_pointLightSize++;
		}
	}
	m_pointLightInfoBlocks.resize(m_renderInfoBlock.m_pointLightSize);
}

void RenderLayer::CollectSpotLights()
{
	const auto scene = GetScene();
	m_renderInfoBlock.m_spotLightSize = 0;
	const std::vector<Entity>* spotLightEntities =
		scene->UnsafeGetPrivateComponentOwnersList<SpotLight>();
	if (spotLightEntities && !spotLightEntities->empty())
	{
		m_spotLightInfoBlocks.resize(spotLightEntities->size());
		for (int i = 0; i < spotLightEntities->size(); i++)
		{
			Entity lightEntity = spotLightEntities->at(i);
			if (!scene->IsEntityEnabled(lightEntity))
				continue;
			const auto slc = scene->GetOrSetPrivateComponent<SpotLight>(lightEntity).lock();
			if (!slc->IsEnabled())
				continue;
			auto ltw = scene->GetDataComponent<GlobalTransform>(lightEntity);
			glm::vec3 position = ltw.m_value[3];
			glm::vec3 front = ltw.GetRotation() * glm::vec3(0, 0, -1);
			glm::vec3 up = ltw.GetRotation() * glm::vec3(0, 1, 0);
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_position = glm::vec4(position, 0);
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_direction = glm::vec4(front, 0);
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_constantLinearQuadFarPlane.x = slc->m_constant;
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_constantLinearQuadFarPlane.y = slc->m_linear;
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_constantLinearQuadFarPlane.z = slc->m_quadratic;
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_constantLinearQuadFarPlane.w = slc->GetFarPlane();
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_diffuse =
				glm::vec4(slc->m_diffuse * slc->m_diffuseBrightness, slc->m_castShadow);
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_specular = glm::vec4(0);

			glm::mat4 shadowProj = glm::perspective(glm::radians(slc->m_outerDegrees * 2.0f), 1.0f,
				1.0f,
				m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_constantLinearQuadFarPlane.w);
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_lightSpaceMatrix = shadowProj * glm::lookAt(position, position + front, up);
			m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_cutOffOuterCutOffLightSizeBias = glm::vec4(
				glm::cos(glm::radians(slc->m_innerDegrees)),
				glm::cos(glm::radians(slc->m_outerDegrees)),
				slc->m_lightSize,
				slc->m_bias);

			switch (m_renderInfoBlock.m_spotLightSize)
			{
			case 0:
				m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_viewPort =
					glm::ivec4(0, 0, Graphics::StorageSizes::m_spotLightShadowMapResolution / 2, Graphics::StorageSizes::m_spotLightShadowMapResolution / 2);
				break;
			case 1:
				m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_viewPort = glm::ivec4(
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					0,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2);
				break;
			case 2:
				m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_viewPort = glm::ivec4(
					0,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2);
				break;
			case 3:
				m_spotLightInfoBlocks[m_renderInfoBlock.m_spotLightSize].m_viewPort = glm::ivec4(
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2,
					Graphics::StorageSizes::m_spotLightShadowMapResolution / 2);
				break;
			}
			m_renderInfoBlock.m_spotLightSize++;
		}
	}
	m_spotLightInfoBlocks.resize(m_renderInfoBlock.m_spotLightSize);
}

void RenderLayer::PreparePointAndSpotLightShadowMap()
{
	const auto& pointLightShadowPipeline = Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP");
	const auto& spotLightShadowPipeline = Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP");

	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GraphicsGlobalStates& globalPipelineState)
		{
#pragma region Viewport and scissor
			VkRect2D renderArea;
			renderArea.offset = { 0, 0 };
			renderArea.extent.width = m_shadowMaps->m_pointLightShadowMap->GetExtent().width;
			renderArea.extent.height = m_shadowMaps->m_pointLightShadowMap->GetExtent().height;

			VkViewport viewport;
			viewport.x = 0;
			viewport.y = 0;
			viewport.width = renderArea.extent.width;
			viewport.height = renderArea.extent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor;
			scissor.offset = { 0, 0 };
			scissor.extent.width = renderArea.extent.width;
			scissor.extent.height = renderArea.extent.height;

			globalPipelineState.m_scissor = scissor;
#pragma endregion
			m_shadowMaps->m_pointLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
			auto depthAttachment = m_shadowMaps->GetPointLightDepthAttachmentInfo();
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 0;
			renderInfo.pColorAttachments = nullptr;
			renderInfo.pDepthAttachment = &depthAttachment;

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			pointLightShadowPipeline->Bind(commandBuffer);
			pointLightShadowPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());

			globalPipelineState.m_colorBlendAttachmentStates.clear();
			globalPipelineState.m_viewPort = viewport;
			globalPipelineState.ApplyAllStates(commandBuffer, true);
			for (int i = 0; i < m_pointLightInfoBlocks.size(); i++)
			{
				const auto& pointLightInfoBlock = m_pointLightInfoBlocks[i];
				viewport.x = pointLightInfoBlock.m_viewPort.x;
				viewport.y = pointLightInfoBlock.m_viewPort.y;
				viewport.width = pointLightInfoBlock.m_viewPort.z;
				viewport.height = pointLightInfoBlock.m_viewPort.w;
				globalPipelineState.m_viewPort = viewport;
				globalPipelineState.ApplyAllStates(commandBuffer);
				m_deferredRenderInstances.Dispatch([&](const std::shared_ptr<Material>& material)
					{}, [&](const RenderInstance& renderCommand)
					{
						switch (renderCommand.m_geometryType)
						{
						case RenderGeometryType::Mesh: {

							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = 0;
							pushConstant.m_materialIndex = i;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							pointLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto mesh = std::dynamic_pointer_cast<Mesh>(renderCommand.m_renderGeometry);
							mesh->Bind(commandBuffer);
							mesh->DrawIndexed(commandBuffer, globalPipelineState);
							break;
						}
						}
					}
				);
			}
			vkCmdEndRendering(commandBuffer);
#pragma region Viewport and scissor

			renderArea.offset = { 0, 0 };
			renderArea.extent.width = m_shadowMaps->m_spotLightShadowMap->GetExtent().width;
			renderArea.extent.height = m_shadowMaps->m_spotLightShadowMap->GetExtent().height;

			viewport.x = 0;
			viewport.y = 0;
			viewport.width = renderArea.extent.width;
			viewport.height = renderArea.extent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			scissor.offset = { 0, 0 };
			scissor.extent.width = renderArea.extent.width;
			scissor.extent.height = renderArea.extent.height;

			globalPipelineState.m_scissor = scissor;
#pragma endregion
			m_shadowMaps->m_spotLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
			depthAttachment = m_shadowMaps->GetSpotLightDepthAttachmentInfo();
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 0;
			renderInfo.pColorAttachments = nullptr;
			renderInfo.pDepthAttachment = &depthAttachment;

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			spotLightShadowPipeline->Bind(commandBuffer);
			spotLightShadowPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());

			globalPipelineState.m_colorBlendAttachmentStates.clear();
			globalPipelineState.m_viewPort = viewport;
			globalPipelineState.ApplyAllStates(commandBuffer, true);
			for (int i = 0; i < m_spotLightInfoBlocks.size(); i++)
			{
				const auto& spotLightInfoBlock = m_spotLightInfoBlocks[i];
				viewport.x = spotLightInfoBlock.m_viewPort.x;
				viewport.y = spotLightInfoBlock.m_viewPort.y;
				viewport.width = spotLightInfoBlock.m_viewPort.z;
				viewport.height = spotLightInfoBlock.m_viewPort.w;
				globalPipelineState.m_viewPort = viewport;
				globalPipelineState.ApplyAllStates(commandBuffer);
				m_deferredRenderInstances.Dispatch([&](const std::shared_ptr<Material>& material)
					{}, [&](const RenderInstance& renderCommand)
					{
						switch (renderCommand.m_geometryType)
						{
						case RenderGeometryType::Mesh: {
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = 0;
							pushConstant.m_materialIndex = i;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							spotLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto mesh = std::dynamic_pointer_cast<Mesh>(renderCommand.m_renderGeometry);
							mesh->Bind(commandBuffer);
							mesh->DrawIndexed(commandBuffer, globalPipelineState);
							break;
						}
						}
					}
				);
			}
			vkCmdEndRendering(commandBuffer);
		});
}

void RenderLayer::CollectRenderInstances(Bound& worldBound)
{
	auto scene = GetScene();

	auto& minBound = worldBound.m_min;
	auto& maxBound = worldBound.m_max;
	minBound = glm::vec3(FLT_MAX);
	maxBound = glm::vec3(FLT_MIN);

	if (const std::vector<Entity>* owners =
		scene->UnsafeGetPrivateComponentOwnersList<MeshRenderer>())
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto mmc = scene->GetOrSetPrivateComponent<MeshRenderer>(owner).lock();
			auto material = mmc->m_material.Get<Material>();
			auto mesh = mmc->m_mesh.Get<Mesh>();
			if (!mmc->IsEnabled() || material == nullptr || mesh == nullptr)
				continue;
			auto gt = scene->GetDataComponent<GlobalTransform>(owner);
			auto ltw = gt.m_value;
			auto meshBound = mesh->GetBound();
			meshBound.ApplyTransform(ltw);
			glm::vec3 center = meshBound.Center();

			glm::vec3 size = meshBound.Size();
			minBound = glm::vec3(
				(glm::min)(minBound.x, center.x - size.x),
				(glm::min)(minBound.y, center.y - size.y),
				(glm::min)(minBound.z, center.z - size.z));
			maxBound = glm::vec3(
				(glm::max)(maxBound.x, center.x + size.x),
				(glm::max)(maxBound.y, center.y + size.y),
				(glm::max)(maxBound.z, center.z + size.z));

			MaterialInfoBlock materialInfoBlock;
			material->UpdateMaterialInfoBlock(materialInfoBlock);
			auto materialIndex = RegisterMaterialIndex(material->GetHandle(), materialInfoBlock);
			InstanceInfoBlock instanceInfoBlock;
			instanceInfoBlock.m_model = gt;
			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);
			RenderInstance renderInstance;
			renderInstance.m_owner = owner;
			renderInstance.m_renderGeometry = mesh;
			renderInstance.m_castShadow = mmc->m_castShadow;
			renderInstance.m_receiveShadow = mmc->m_receiveShadow;
			renderInstance.m_geometryType = RenderGeometryType::Mesh;

			renderInstance.m_materialIndex = materialIndex;
			renderInstance.m_instanceIndex = instanceIndex;

			if (material->m_drawSettings.m_blending)
			{
				auto& group = m_transparentRenderInstances.m_renderInstanceGroups[material->GetHandle()];
				group.m_material = material;
				group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);

			}
			else if (mmc->m_forwardRendering)
			{
				auto& group = m_forwardRenderInstances.m_renderInstanceGroups[material->GetHandle()];
				group.m_material = material;
				group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
			}
			else
			{
				auto& group = m_deferredRenderInstances.m_renderInstanceGroups[material->GetHandle()];
				group.m_material = material;
				group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
			}
		}
	}
	/*
	owners = scene->UnsafeGetPrivateComponentOwnersList<Particles>();
	if (owners)
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto particles = scene->GetOrSetPrivateComponent<Particles>(owner).lock();
			auto material = particles->m_material.Get<Material>();
			auto mesh = particles->m_mesh.Get<Mesh>();
			if (!particles->IsEnabled() || material == nullptr || mesh == nullptr)
				continue;
			auto gt = scene->GetDataComponent<GlobalTransform>(owner);
			auto ltw = gt.m_value;
			auto meshBound = mesh->GetBound();
			meshBound.ApplyTransform(ltw);
			glm::vec3 center = meshBound.Center();

			glm::vec3 size = meshBound.Size();
			minBound = glm::vec3(
				(glm::min)(minBound.x, center.x - size.x),
				(glm::min)(minBound.y, center.y - size.y),
				(glm::min)(minBound.z, center.z - size.z));

			maxBound = glm::vec3(
				(glm::max)(maxBound.x, center.x + size.x),
				(glm::max)(maxBound.y, center.y + size.y),
				(glm::max)(maxBound.z, center.z + size.z));
			for (const auto& pair : cameraPairs)
			{
				auto& deferredRenderInstances = m_deferredRenderInstances[pair.first->GetHandle()];
				auto& deferredInstancedRenderInstances = m_deferredInstancedRenderInstances[pair.first->GetHandle()];
				auto& forwardRenderInstances = m_forwardRenderInstances[pair.first->GetHandle()];
				auto& forwardInstancedRenderInstances = m_forwardInstancedRenderInstances[pair.first->GetHandle()];
				auto& transparentRenderInstances = m_transparentRenderInstances[pair.first->GetHandle()];
				auto& instancedTransparentRenderInstances = m_instancedTransparentRenderInstances[pair.first->GetHandle()];

				deferredRenderInstances.m_camera = pair.first;
				deferredInstancedRenderInstances.m_camera = pair.first;
				forwardRenderInstances.m_camera = pair.first;
				forwardInstancedRenderInstances.m_camera = pair.first;
				transparentRenderInstances.m_camera = pair.first;
				instancedTransparentRenderInstances.m_camera = pair.first;

				RenderInstance renderInstance;
				renderInstance.m_owner = owner;
				renderInstance.m_globalTransform = gt;
				renderInstance.m_renderGeometry = mesh;
				renderInstance.m_castShadow = particles->m_castShadow;
				renderInstance.m_receiveShadow = particles->m_receiveShadow;
				renderInstance.m_matrices = particles->m_matrices;
				renderInstance.m_geometryType = RenderGeometryType::Mesh;
				if (material->m_drawSettings.m_blending)
				{
					auto& group = instancedTransparentRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
				}
				else if (particles->m_forwardRendering)
				{
					auto& group = forwardInstancedRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
				}
				else
				{
					auto& group = deferredInstancedRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
				}
			}
		}
	}
	owners = scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>();
	if (owners)
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto smmc = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(owner).lock();
			auto material = smmc->m_material.Get<Material>();
			auto skinnedMesh = smmc->m_skinnedMesh.Get<SkinnedMesh>();
			if (!smmc->IsEnabled() || material == nullptr || skinnedMesh == nullptr)
				continue;
			GlobalTransform gt;
			auto animator = smmc->m_animator.Get<Animator>();
			if (!animator)
			{
				continue;
			}
			if (!smmc->m_ragDoll)
			{
				gt = scene->GetDataComponent<GlobalTransform>(owner);
			}
			auto ltw = gt.m_value;
			auto meshBound = skinnedMesh->GetBound();
			meshBound.ApplyTransform(ltw);
			glm::vec3 center = meshBound.Center();

			glm::vec3 size = meshBound.Size();
			minBound = glm::vec3(
				(glm::min)(minBound.x, center.x - size.x),
				(glm::min)(minBound.y, center.y - size.y),
				(glm::min)(minBound.z, center.z - size.z));
			maxBound = glm::vec3(
				(glm::max)(maxBound.x, center.x + size.x),
				(glm::max)(maxBound.y, center.y + size.y),
				(glm::max)(maxBound.z, center.z + size.z));
			for (const auto& pair : cameraPairs)
			{
				auto& deferredRenderInstances = m_deferredRenderInstances[pair.first->GetHandle()];
				auto& deferredInstancedRenderInstances = m_deferredInstancedRenderInstances[pair.first->GetHandle()];
				auto& forwardRenderInstances = m_forwardRenderInstances[pair.first->GetHandle()];
				auto& forwardInstancedRenderInstances = m_forwardInstancedRenderInstances[pair.first->GetHandle()];
				auto& transparentRenderInstances = m_transparentRenderInstances[pair.first->GetHandle()];
				auto& instancedTransparentRenderInstances = m_instancedTransparentRenderInstances[pair.first->GetHandle()];

				deferredRenderInstances.m_camera = pair.first;
				deferredInstancedRenderInstances.m_camera = pair.first;
				forwardRenderInstances.m_camera = pair.first;
				forwardInstancedRenderInstances.m_camera = pair.first;
				transparentRenderInstances.m_camera = pair.first;
				instancedTransparentRenderInstances.m_camera = pair.first;

				RenderInstance renderInstance;
				renderInstance.m_owner = owner;
				renderInstance.m_globalTransform = gt;
				renderInstance.m_renderGeometry = skinnedMesh;
				renderInstance.m_castShadow = smmc->m_castShadow;
				renderInstance.m_receiveShadow = smmc->m_receiveShadow;
				renderInstance.m_geometryType = RenderGeometryType::SkinnedMesh;
				renderInstance.m_boneMatrices = smmc->m_finalResults;
				if (material->m_drawSettings.m_blending)
				{
					auto& group = transparentRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[skinnedMesh->GetHandle()].push_back(renderInstance);
				}
				else if (smmc->m_forwardRendering)
				{
					auto& group = forwardRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[skinnedMesh->GetHandle()].push_back(renderInstance);
				}
				else
				{
					auto& group = deferredRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[skinnedMesh->GetHandle()].push_back(renderInstance);
				}
			}
		}
	}
	owners =
		scene->UnsafeGetPrivateComponentOwnersList<StrandsRenderer>();
	if (owners)
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto mmc = scene->GetOrSetPrivateComponent<StrandsRenderer>(owner).lock();
			auto material = mmc->m_material.Get<Material>();
			auto strands = mmc->m_strands.Get<Strands>();
			if (!mmc->IsEnabled() || material == nullptr || strands == nullptr)
				continue;
			auto gt = scene->GetDataComponent<GlobalTransform>(owner);
			auto ltw = gt.m_value;
			auto meshBound = strands->m_bound;
			meshBound.ApplyTransform(ltw);
			glm::vec3 center = meshBound.Center();

			glm::vec3 size = meshBound.Size();
			minBound = glm::vec3(
				(glm::min)(minBound.x, center.x - size.x),
				(glm::min)(minBound.y, center.y - size.y),
				(glm::min)(minBound.z, center.z - size.z));
			maxBound = glm::vec3(
				(glm::max)(maxBound.x, center.x + size.x),
				(glm::max)(maxBound.y, center.y + size.y),
				(glm::max)(maxBound.z, center.z + size.z));

			auto meshCenter = gt.m_value * glm::vec4(center, 1.0);
			for (const auto& pair : cameraPairs)
			{
				auto& deferredRenderInstances = m_deferredRenderInstances[pair.first->GetHandle()];
				auto& deferredInstancedRenderInstances = m_deferredInstancedRenderInstances[pair.first->GetHandle()];
				auto& forwardRenderInstances = m_forwardRenderInstances[pair.first->GetHandle()];
				auto& forwardInstancedRenderInstances = m_forwardInstancedRenderInstances[pair.first->GetHandle()];
				auto& transparentRenderInstances = m_transparentRenderInstances[pair.first->GetHandle()];
				auto& instancedTransparentRenderInstances = m_instancedTransparentRenderInstances[pair.first->GetHandle()];

				deferredRenderInstances.m_camera = pair.first;
				deferredInstancedRenderInstances.m_camera = pair.first;
				forwardRenderInstances.m_camera = pair.first;
				forwardInstancedRenderInstances.m_camera = pair.first;
				transparentRenderInstances.m_camera = pair.first;
				instancedTransparentRenderInstances.m_camera = pair.first;

				RenderInstance renderInstance;
				renderInstance.m_owner = owner;
				renderInstance.m_globalTransform = gt;
				renderInstance.m_renderGeometry = strands;
				renderInstance.m_castShadow = mmc->m_castShadow;
				renderInstance.m_receiveShadow = mmc->m_receiveShadow;
				renderInstance.m_geometryType = RenderGeometryType::Strands;
				if (material->m_drawSettings.m_blending)
				{
					auto& group = transparentRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[strands->GetHandle()].push_back(renderInstance);
				}
				else if (mmc->m_forwardRendering)
				{
					auto& group = forwardRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[strands->GetHandle()].push_back(renderInstance);
				}
				else
				{
					auto& group = deferredRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[strands->GetHandle()].push_back(renderInstance);
				}
			}
		}
	}
	*/
}




void RenderLayer::CreateStandardDescriptorBuffers()
{
#pragma region Standard Descrioptor Layout
	m_renderInfoDescriptorBuffers.clear();
	m_environmentInfoDescriptorBuffers.clear();
	m_cameraInfoDescriptorBuffers.clear();
	m_materialInfoDescriptorBuffers.clear();
	m_objectInfoDescriptorBuffers.clear();
	m_kernelDescriptorBuffers.clear();
	m_directionalLightInfoDescriptorBuffers.clear();
	m_pointLightInfoDescriptorBuffers.clear();
	m_spotLightInfoDescriptorBuffers.clear();

	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo bufferVmaAllocationCreateInfo{};
	bufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	bufferVmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	m_renderInfoBlockMemory.resize(maxFrameInFlight);
	m_environmentalInfoBlockMemory.resize(maxFrameInFlight);
	m_cameraInfoBlockMemory.resize(maxFrameInFlight);
	m_materialInfoBlockMemory.resize(maxFrameInFlight);
	m_instanceInfoBlockMemory.resize(maxFrameInFlight);
	m_kernelBlockMemory.resize(maxFrameInFlight);
	m_directionalLightInfoBlockMemory.resize(maxFrameInFlight);
	m_pointLightInfoBlockMemory.resize(maxFrameInFlight);
	m_spotLightInfoBlockMemory.resize(maxFrameInFlight);
	for (size_t i = 0; i < maxFrameInFlight; i++) {
		bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(RenderInfoBlock);
		m_renderInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(EnvironmentInfoBlock);
		m_environmentInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(CameraInfoBlock) * Graphics::StorageSizes::m_maxCameraSize;
		m_cameraInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(MaterialInfoBlock) * Graphics::StorageSizes::m_maxMaterialSize;
		m_materialInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(InstanceInfoBlock) * Graphics::StorageSizes::m_maxInstanceSize;
		m_objectInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(glm::vec4) * Graphics::StorageSizes::m_maxKernelAmount * 2;
		m_kernelDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(DirectionalLightInfo) * Graphics::StorageSizes::m_maxDirectionalLightSize * Graphics::StorageSizes::m_maxCameraSize;
		m_directionalLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(PointLightInfo) * Graphics::StorageSizes::m_maxPointLightSize;
		m_pointLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(SpotLightInfo) * Graphics::StorageSizes::m_maxSpotLightSize;
		m_spotLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));


		vmaMapMemory(Graphics::GetVmaAllocator(), m_renderInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_renderInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_environmentInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_environmentalInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_cameraInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_cameraInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_materialInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_materialInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_objectInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_instanceInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_kernelDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_kernelBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_directionalLightInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_directionalLightInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_pointLightInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_pointLightInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_spotLightInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_spotLightInfoBlockMemory[i])));
	}
#pragma endregion
}



void RenderLayer::UpdateStandardBindings()
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	
	m_perFrameDescriptorSets.clear();
	for(size_t i = 0; i < maxFramesInFlight; i++)
	{
		auto descriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("PER_FRAME_LAYOUT"));
		
		VkDescriptorBufferInfo bufferInfo;
		bufferInfo.offset = 0;
		bufferInfo.range = VK_WHOLE_SIZE;

		bufferInfo.buffer = m_renderInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(0, bufferInfo);
		bufferInfo.buffer = m_environmentInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(1, bufferInfo);
		bufferInfo.buffer = m_cameraInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(2, bufferInfo);
		bufferInfo.buffer = m_materialInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(3, bufferInfo);
		bufferInfo.buffer = m_objectInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(4, bufferInfo);
		bufferInfo.buffer = m_kernelDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(6, bufferInfo);
		bufferInfo.buffer = m_directionalLightInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(7, bufferInfo);
		bufferInfo.buffer = m_pointLightInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(8, bufferInfo);
		bufferInfo.buffer = m_spotLightInfoDescriptorBuffers[i]->GetVkBuffer();
		descriptorSet->UpdateBufferDescriptorBinding(9, bufferInfo);
		m_perFrameDescriptorSets.emplace_back(descriptorSet);
	}
}

void RenderLayer::PrepareEnvironmentalBrdfLut()
{
	m_environmentalBRDFSampler.reset();
	m_environmentalBRDFView.reset();
	m_environmentalBRDFLut.reset();
	auto brdfLutResolution = 512;
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = brdfLutResolution;
		imageInfo.extent.height = brdfLutResolution;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = VK_FORMAT_R16G16_SFLOAT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_environmentalBRDFLut = std::make_shared<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_environmentalBRDFLut->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_R16G16_SFLOAT;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_environmentalBRDFView = std::make_shared<ImageView>(viewInfo);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		m_environmentalBRDFSampler = std::make_shared<Sampler>(samplerInfo);
	}
	
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer, GraphicsGlobalStates& globalPipelineState) {
		m_environmentalBRDFLut->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
#pragma region Viewport and scissor
		VkRect2D renderArea;
		renderArea.offset = { 0, 0 };
		renderArea.extent.width = brdfLutResolution;
		renderArea.extent.height = brdfLutResolution;
		VkViewport viewport;
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = brdfLutResolution;
		viewport.height = brdfLutResolution;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent.width = brdfLutResolution;
		scissor.extent.height = brdfLutResolution;
		globalPipelineState.m_viewPort = viewport;
		globalPipelineState.m_scissor = scissor;
#pragma endregion
#pragma region Lighting pass
		{
			VkRenderingAttachmentInfo attachment{};
			attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

			attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			attachment.clearValue = { 0, 0, 0, 1 };
			attachment.imageView = m_environmentalBRDFView->GetVkImageView();

			//const auto depthAttachment = camera->GetRenderTexture()->GetDepthAttachmentInfo();
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 1;
			renderInfo.pColorAttachments = &attachment;
			//renderInfo.pDepthAttachment = &depthAttachment;
			
			
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			Graphics::GetGraphicsPipeline("ENVIRONMENTAL_MAP_BRDF")->Bind(commandBuffer);
			globalPipelineState.m_depthTest = false;
			globalPipelineState.m_colorBlendAttachmentStates.clear();
			globalPipelineState.m_colorBlendAttachmentStates.resize(1);
			for (auto& i : globalPipelineState.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
				i.blendEnable = VK_FALSE;
			}
			globalPipelineState.ApplyAllStates(commandBuffer, true);
			const auto mesh = std::dynamic_pointer_cast<Mesh>(Resources::GetResource("PRIMITIVE_TEX_PASS_THROUGH"));
			mesh->Bind(commandBuffer);
			mesh->DrawIndexed(commandBuffer, globalPipelineState);
			vkCmdEndRendering(commandBuffer);
#pragma endregion
		}
		m_environmentalBRDFLut->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	);
}

void RenderLayer::PrepareDescriptorSetLayouts()
{
	const auto perFrameLayout = std::make_shared<DescriptorSetLayout>();
	perFrameLayout->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->PushDescriptorBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	perFrameLayout->Initialize();
	Graphics::RegisterDescriptorSetLayout("PER_FRAME_LAYOUT", perFrameLayout);

	const auto pbrTextureLayout = std::make_shared<DescriptorSetLayout>();
	pbrTextureLayout->PushDescriptorBinding(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	pbrTextureLayout->PushDescriptorBinding(11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	pbrTextureLayout->PushDescriptorBinding(12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	pbrTextureLayout->PushDescriptorBinding(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	pbrTextureLayout->PushDescriptorBinding(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	pbrTextureLayout->Initialize();
	Graphics::RegisterDescriptorSetLayout("PBR_TEXTURE_LAYOUT", pbrTextureLayout);

	const auto cameraGBufferLayout = std::make_shared<DescriptorSetLayout>();
	cameraGBufferLayout->PushDescriptorBinding(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	cameraGBufferLayout->PushDescriptorBinding(11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	cameraGBufferLayout->PushDescriptorBinding(12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	cameraGBufferLayout->PushDescriptorBinding(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	cameraGBufferLayout->Initialize();
	Graphics::RegisterDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT", cameraGBufferLayout);

	const auto lightLayout = std::make_shared<DescriptorSetLayout>();
	lightLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	lightLayout->PushDescriptorBinding(19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	lightLayout->PushDescriptorBinding(20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	lightLayout->PushDescriptorBinding(21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	lightLayout->Initialize();
	Graphics::RegisterDescriptorSetLayout("LIGHTING_LAYOUT", lightLayout);\

	const auto equirectangularToCubeLayout = std::make_shared<DescriptorSetLayout>();
	equirectangularToCubeLayout->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
	equirectangularToCubeLayout->Initialize();
	Graphics::RegisterDescriptorSetLayout("EQUIRECTANGULAR_TO_CUBE_LAYOUT", equirectangularToCubeLayout);
}



void RenderLayer::RenderToCamera(const std::shared_ptr<Camera>& camera)
{
	auto cameraIndex = GetCameraIndex(camera->GetHandle());

	const auto& directionalLightShadowPipeline = Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP");

	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GraphicsGlobalStates& globalPipelineState)
		{
#pragma region Viewport and scissor
			VkRect2D renderArea;
			renderArea.offset = { 0, 0 };
			renderArea.extent.width = m_shadowMaps->m_directionalLightShadowMap->GetExtent().width;
			renderArea.extent.height = m_shadowMaps->m_directionalLightShadowMap->GetExtent().height;

			VkViewport viewport;
			viewport.x = 0;
			viewport.y = 0;
			viewport.width = renderArea.extent.width;
			viewport.height = renderArea.extent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor;
			scissor.offset = { 0, 0 };
			scissor.extent.width = renderArea.extent.width;
			scissor.extent.height = renderArea.extent.height;

			globalPipelineState.m_scissor = scissor;
#pragma endregion
			m_shadowMaps->m_directionalLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
			const auto depthAttachment = m_shadowMaps->GetDirectionalLightDepthAttachmentInfo();
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 0;
			renderInfo.pColorAttachments = nullptr;
			renderInfo.pDepthAttachment = &depthAttachment;

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			directionalLightShadowPipeline->Bind(commandBuffer);
			directionalLightShadowPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			globalPipelineState.m_viewPort = viewport;
			globalPipelineState.m_colorBlendAttachmentStates.clear();
			globalPipelineState.ApplyAllStates(commandBuffer, true);
			for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
			{
				const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::StorageSizes::m_maxDirectionalLightSize + i];
				viewport.x = directionalLightInfoBlock.m_viewPort.x;
				viewport.y = directionalLightInfoBlock.m_viewPort.y;
				viewport.width = directionalLightInfoBlock.m_viewPort.z;
				viewport.height = directionalLightInfoBlock.m_viewPort.w;
				globalPipelineState.m_viewPort = viewport;
				globalPipelineState.ApplyAllStates(commandBuffer);
				m_deferredRenderInstances.Dispatch([&](const std::shared_ptr<Material>& material)
					{}, [&](const RenderInstance& renderCommand)
					{
						switch (renderCommand.m_geometryType)
						{
						case RenderGeometryType::Mesh: {

							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = cameraIndex;
							pushConstant.m_materialIndex = cameraIndex * Graphics::StorageSizes::m_maxDirectionalLightSize + i;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							directionalLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto mesh = std::dynamic_pointer_cast<Mesh>(renderCommand.m_renderGeometry);
							mesh->Bind(commandBuffer);
							mesh->DrawIndexed(commandBuffer, globalPipelineState);
							break;
						}
						}
					}
				);
			}
			vkCmdEndRendering(commandBuffer);
		}
	);


	const auto& deferredPrepassPipeline = Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_PREPASS");
	const auto& deferredLightingPipeline = Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_LIGHTING");
	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GraphicsGlobalStates& globalPipelineState) {
#pragma region Viewport and scissor
		VkRect2D renderArea;
		renderArea.offset = { 0, 0 };
		renderArea.extent.width = camera->GetSize().x;
		renderArea.extent.height = camera->GetSize().y;
		VkViewport viewport;
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = camera->GetSize().x;
		viewport.height = camera->GetSize().y;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent.width = camera->GetSize().x;
		scissor.extent.height = camera->GetSize().y;
		globalPipelineState.m_viewPort = viewport;
		globalPipelineState.m_scissor = scissor;
#pragma endregion
#pragma region Geometry pass
		{
			camera->TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos);
			const auto depthAttachment = camera->GetDepthAttachmentInfo();
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();
			renderInfo.pDepthAttachment = &depthAttachment;

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredPrepassPipeline->Bind(commandBuffer);
			deferredPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());

			globalPipelineState.m_colorBlendAttachmentStates.clear();
			globalPipelineState.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
			for (auto& i : globalPipelineState.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				i.blendEnable = VK_FALSE;
			}
			globalPipelineState.ApplyAllStates(commandBuffer, true);

			m_deferredRenderInstances.Dispatch([&](const std::shared_ptr<Material>& material)
				{
					deferredPrepassPipeline->BindDescriptorSet(commandBuffer, 1, material->m_descriptorSet->GetVkDescriptorSet());
					//We should also bind textures here.
				}, [&](const RenderInstance& renderCommand)
				{
					switch (renderCommand.m_geometryType)
					{
					case RenderGeometryType::Mesh: {
						globalPipelineState.ApplyAllStates(commandBuffer);
						RenderInstancePushConstant pushConstant;
						pushConstant.m_cameraIndex = cameraIndex;
						pushConstant.m_materialIndex = renderCommand.m_materialIndex;
						pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
						deferredPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
						const auto mesh = std::dynamic_pointer_cast<Mesh>(renderCommand.m_renderGeometry);
						mesh->Bind(commandBuffer);
						mesh->DrawIndexed(commandBuffer, globalPipelineState);
						break;
					}
					}
				}
				);

			vkCmdEndRendering(commandBuffer);
		}
#pragma endregion

#pragma region Lighting pass
		{
			camera->TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->GetRenderTexture()->AppendColorAttachmentInfos(colorAttachmentInfos);
			const auto depthAttachment = camera->GetRenderTexture()->GetDepthAttachmentInfo();
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();
			renderInfo.pDepthAttachment = &depthAttachment;
			m_shadowMaps->m_directionalLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			m_shadowMaps->m_pointLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			m_shadowMaps->m_spotLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredLightingPipeline->Bind(commandBuffer);
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 1, camera->m_gBufferDescriptorSet->GetVkDescriptorSet());
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 2, m_shadowMaps->m_lightingDescriptorSet->GetVkDescriptorSet());
			globalPipelineState.m_depthTest = false;
			globalPipelineState.m_colorBlendAttachmentStates.clear();
			globalPipelineState.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
			for (auto& i : globalPipelineState.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				i.blendEnable = VK_FALSE;
			}
			globalPipelineState.ApplyAllStates(commandBuffer, true);
			RenderInstancePushConstant pushConstant;
			pushConstant.m_cameraIndex = cameraIndex;
			pushConstant.m_materialIndex = 0;
			pushConstant.m_instanceIndex = 0;
			deferredLightingPipeline->PushConstant(commandBuffer, 0, pushConstant);
			const auto mesh = std::dynamic_pointer_cast<Mesh>(Resources::GetResource("PRIMITIVE_TEX_PASS_THROUGH"));
			mesh->Bind(commandBuffer);
			mesh->DrawIndexed(commandBuffer, globalPipelineState, false);
			vkCmdEndRendering(commandBuffer);
		}
#pragma endregion
		}
	);


	camera->m_rendered = true;
	camera->m_requireRendering = false;
}


void RenderLayer::UploadEnvironmentalInfoBlock(const EnvironmentInfoBlock& environmentInfoBlock) const
{
	memcpy(m_environmentalInfoBlockMemory[Graphics::GetCurrentFrameIndex()], &environmentInfoBlock, sizeof(EnvironmentInfoBlock));
}

void RenderLayer::UploadRenderInfoBlock(const RenderInfoBlock& renderInfoBlock) const
{
	memcpy(m_renderInfoBlockMemory[Graphics::GetCurrentFrameIndex()], &renderInfoBlock, sizeof(RenderInfoBlock));
}


void RenderLayer::UploadDirectionalLightInfoBlocks(const std::vector<DirectionalLightInfo>& directionalLightInfoBlocks) const
{
	memcpy(m_directionalLightInfoBlockMemory[Graphics::GetCurrentFrameIndex()], directionalLightInfoBlocks.data(), sizeof(DirectionalLightInfo) * directionalLightInfoBlocks.size());
}

void RenderLayer::UploadPointLightInfoBlocks(const std::vector<PointLightInfo>& pointLightInfoBlocks) const
{
	memcpy(m_pointLightInfoBlockMemory[Graphics::GetCurrentFrameIndex()], pointLightInfoBlocks.data(), sizeof(PointLightInfo) * pointLightInfoBlocks.size());
}

void RenderLayer::UploadSpotLightInfoBlocks(const std::vector<SpotLightInfo>& spotLightInfoBlocks) const
{
	memcpy(m_spotLightInfoBlockMemory[Graphics::GetCurrentFrameIndex()], spotLightInfoBlocks.data(), sizeof(SpotLightInfo) * spotLightInfoBlocks.size());
}

void RenderLayer::UploadCameraInfoBlocks(const std::vector<CameraInfoBlock>& cameraInfoBlocks) const
{
	memcpy(m_cameraInfoBlockMemory[Graphics::GetCurrentFrameIndex()], cameraInfoBlocks.data(), sizeof(CameraInfoBlock) * cameraInfoBlocks.size());
}

void RenderLayer::UploadMaterialInfoBlocks(const std::vector<MaterialInfoBlock>& materialInfoBlocks) const
{
	memcpy(m_materialInfoBlockMemory[Graphics::GetCurrentFrameIndex()], materialInfoBlocks.data(), sizeof(MaterialInfoBlock) * materialInfoBlocks.size());
}

void RenderLayer::UploadInstanceInfoBlocks(const std::vector<InstanceInfoBlock>& objectInfoBlocks) const
{
	memcpy(m_instanceInfoBlockMemory[Graphics::GetCurrentFrameIndex()], objectInfoBlocks.data(), sizeof(InstanceInfoBlock) * objectInfoBlocks.size());
}


