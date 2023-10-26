#include "RenderLayer.hpp"
#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "ProjectManager.hpp"
#include "Particles.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "GraphicsPipeline.hpp"
#include "PostProcessingStack.hpp"
#include "Jobs.hpp"
#include "GeometryStorage.hpp"
#include "PostProcessingStack.hpp"
#include "StrandsRenderer.hpp"
#include "TextureStorage.hpp"
using namespace EvoEngine;

void RenderInstanceCollection::Dispatch(const std::function<void(const RenderInstance&)>& commandAction) const
{
	for (const auto& renderCommand : m_renderCommands)
	{
		commandAction(renderCommand);
	}
}

void SkinnedRenderInstanceCollection::Dispatch(const std::function<void(const SkinnedRenderInstance&)>& commandAction) const
{
	for (const auto& renderCommand : m_renderCommands)
	{
		commandAction(renderCommand);
	}
}

void StrandsRenderInstanceCollection::Dispatch(
	const std::function<void(const StrandsRenderInstance&)>& commandAction) const
{
	for (const auto& renderCommand : m_renderCommands)
	{
		commandAction(renderCommand);
	}
}

void InstancedRenderInstanceCollection::Dispatch(
	const std::function<void(const InstancedRenderInstance&)>& commandAction) const
{
	for (const auto& renderCommand : m_renderCommands)
	{
		commandAction(renderCommand);
	}
}

void RenderLayer::OnCreate()
{
	CreateStandardDescriptorBuffers();
	CreatePerFrameDescriptorSets();

	std::vector<glm::vec4> kernels;
	for (uint32_t i = 0; i < Graphics::Constants::MAX_KERNEL_AMOUNT; i++)
	{
		kernels.emplace_back(glm::ballRand(1.0f), 1.0f);
	}
	for (uint32_t i = 0; i < Graphics::Constants::MAX_KERNEL_AMOUNT; i++)
	{
		kernels.emplace_back(
			glm::gaussRand(0.0f, 1.0f),
			glm::gaussRand(0.0f, 1.0f),
			glm::gaussRand(0.0f, 1.0f),
			glm::gaussRand(0.0f, 1.0f));
	}
	for (int i = 0; i < Graphics::GetMaxFramesInFlight(); i++) {
		m_kernelDescriptorBuffers[i]->UploadVector(kernels);
	}

	PrepareEnvironmentalBrdfLut();

	m_lighting = std::make_unique<Lighting>();
	m_lighting->Initialize();
}

void RenderLayer::OnDestroy()
{
	m_renderInfoDescriptorBuffers.clear();
	m_environmentInfoDescriptorBuffers.clear();
	m_cameraInfoDescriptorBuffers.clear();
	m_materialInfoDescriptorBuffers.clear();
	m_instanceInfoDescriptorBuffers.clear();
}

void RenderLayer::PreUpdate()
{

	m_deferredRenderInstances.m_renderCommands.clear();
	m_deferredSkinnedRenderInstances.m_renderCommands.clear();
	m_deferredInstancedRenderInstances.m_renderCommands.clear();
	m_deferredStrandsRenderInstances.m_renderCommands.clear();
	m_transparentRenderInstances.m_renderCommands.clear();
	m_transparentSkinnedRenderInstances.m_renderCommands.clear();
	m_transparentInstancedRenderInstances.m_renderCommands.clear();
	m_transparentStrandsRenderInstances.m_renderCommands.clear();

	m_cameraIndices.clear();
	m_materialIndices.clear();
	m_instanceIndices.clear();
	m_instanceHandles.clear();
	m_cameraInfoBlocks.clear();
	m_materialInfoBlocks.clear();
	m_instanceInfoBlocks.clear();


	m_directionalLightInfoBlocks.clear();
	m_pointLightInfoBlocks.clear();
	m_spotLightInfoBlocks.clear();

	m_meshDrawIndexedIndirectCommands.clear();
	m_meshDrawMeshTasksIndirectCommands.clear();

	m_cameras.clear();

	const auto scene = GetScene();
	if (!scene) return;

	CollectCameras(m_cameras);
	Bound worldBound;
	m_totalMeshTriangles = 0;

	CollectRenderInstances(worldBound);
	scene->SetBound(worldBound);

	CollectDirectionalLights(m_cameras);
	CollectPointLights();
	CollectSpotLights();
}

void RenderLayer::LateUpdate()
{
	const auto scene = GetScene();
	if (!scene) return;

	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
		{
			for (const auto& i : m_cameras)
			{
				i.second->GetRenderTexture()->Clear(commandBuffer);
			}
		}
	);

	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	auto& graphics = Graphics::GetInstance();
	graphics.m_triangles[currentFrameIndex] = 0;
	graphics.m_strandsSegments[currentFrameIndex] = 0;
	graphics.m_drawCall[currentFrameIndex] = 0;

	GeometryStorage::DeviceSync();
	TextureStorage::DeviceSync();

	ApplyAnimator();
	//The following data stays consistent during entire frame.
	{
		for (int split = 0; split < 4; split++)
		{
			float splitEnd = m_maxShadowDistance;
			if (split != 3)
				splitEnd = m_maxShadowDistance * m_shadowCascadeSplit[split];
			m_renderInfoBlock.m_splitDistances[split] = splitEnd;
		}
		m_renderInfoBlock.m_brdflutTextureIndex = m_environmentalBRDFLut->GetTextureStorageIndex();
	}

	{
		switch (scene->m_environment.m_environmentType)
		{

		case EnvironmentType::EnvironmentalMap: {
			m_environmentInfoBlock.m_backgroundColor.w = 0.0f;
		}
											  break;
		case EnvironmentType::Color: {
			m_environmentInfoBlock.m_backgroundColor = glm::vec4(scene->m_environment.m_backgroundColor, 1.0f);
		}
								   break;
		}
		m_environmentInfoBlock.m_environmentalMapGamma = scene->m_environment.m_environmentGamma;
		m_environmentInfoBlock.m_environmentalLightingIntensity = scene->m_environment.m_ambientLightIntensity;
		m_environmentInfoBlock.m_backgroundIntensity = scene->m_environment.m_backgroundIntensity;
	}

	m_renderInfoDescriptorBuffers[currentFrameIndex]->Upload(m_renderInfoBlock);
	m_environmentInfoDescriptorBuffers[currentFrameIndex]->Upload(m_environmentInfoBlock);
	m_directionalLightInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_directionalLightInfoBlocks);
	m_pointLightInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_pointLightInfoBlocks);
	m_spotLightInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_spotLightInfoBlocks);

	m_cameraInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_cameraInfoBlocks);
	m_materialInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_materialInfoBlocks);
	m_instanceInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_instanceInfoBlocks);

	m_meshDrawIndexedIndirectCommandsBuffers[currentFrameIndex]->UploadVector(m_meshDrawIndexedIndirectCommands);
	m_meshDrawMeshTasksIndirectCommandsBuffers[currentFrameIndex]->UploadVector(m_meshDrawMeshTasksIndirectCommands);


	VkDescriptorBufferInfo bufferInfo;
	bufferInfo.offset = 0;
	bufferInfo.range = VK_WHOLE_SIZE;

	bufferInfo.buffer = m_renderInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(0, bufferInfo);
	bufferInfo.buffer = m_environmentInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(1, bufferInfo);
	bufferInfo.buffer = m_cameraInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(2, bufferInfo);
	bufferInfo.buffer = m_materialInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(3, bufferInfo);
	bufferInfo.buffer = m_instanceInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(4, bufferInfo);
	bufferInfo.buffer = m_kernelDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(5, bufferInfo);
	bufferInfo.buffer = m_directionalLightInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(6, bufferInfo);
	bufferInfo.buffer = m_pointLightInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(7, bufferInfo);
	bufferInfo.buffer = m_spotLightInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(8, bufferInfo);

	bufferInfo.buffer = GeometryStorage::GetVertexBuffer()->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(9, bufferInfo);
	bufferInfo.buffer = GeometryStorage::GetMeshletBuffer()->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(10, bufferInfo);

	PreparePointAndSpotLightShadowMap();

	for (const auto& [cameraGlobalTransform, camera] : m_cameras)
	{
		camera->m_rendered = false;
		if (camera->m_requireRendering)
		{
			RenderToCamera(cameraGlobalTransform, camera);
		}
	}
}

void RenderLayer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("View"))
		{
			ImGui::Checkbox("Render Settings", &m_enableRenderMenu);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if (m_enableRenderMenu)
	{
		ImGui::Begin("Render Settings");
		ImGui::Checkbox("Count drawcall for shadows", &m_countShadowRenderingDrawCalls);
		ImGui::Checkbox("Indirect Rendering", &m_enableIndirectRendering);
		if(Graphics::Constants::ENABLE_MESH_SHADER) ImGui::Checkbox("Meshlet", &Graphics::Settings::USE_MESH_SHADER);
		const bool useMeshShader = Graphics::Constants::ENABLE_MESH_SHADER && Graphics::Settings::USE_MESH_SHADER;
		if (useMeshShader) {
			ImGui::Checkbox("Show meshlets", &m_enableDebugVisualization);
		}else
		{
			ImGui::Checkbox("Show meshes", &m_enableDebugVisualization);
		}
		ImGui::DragFloat("Gamma", &m_renderInfoBlock.m_gamma, 0.01f, 1.0f, 3.0f);
		if (ImGui::CollapsingHeader("Shadow", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::TreeNode("Distance"))
			{
				ImGui::DragFloat("Max shadow distance", &m_maxShadowDistance, 1.0f, 0.1f);
				ImGui::DragFloat("Split 1", &m_shadowCascadeSplit[0], 0.01f, 0.0f, m_shadowCascadeSplit[1]);
				ImGui::DragFloat(
					"Split 2", &m_shadowCascadeSplit[1], 0.01f, m_shadowCascadeSplit[0], m_shadowCascadeSplit[2]);
				ImGui::DragFloat(
					"Split 3", &m_shadowCascadeSplit[2], 0.01f, m_shadowCascadeSplit[1], m_shadowCascadeSplit[3]);
				ImGui::DragFloat("Split 4", &m_shadowCascadeSplit[3], 0.01f, m_shadowCascadeSplit[2], 1.0f);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("PCSS"))
			{
				ImGui::DragInt("Blocker search side amount", &m_renderInfoBlock.m_blockerSearchAmount, 1, 1, 8);
				ImGui::DragInt("PCF Sample Size", &m_renderInfoBlock.m_pcfSampleAmount, 1, 1, 64);
				ImGui::TreePop();
			}
			ImGui::DragFloat("Seam fix ratio", &m_renderInfoBlock.m_seamFixRatio, 0.001f, 0.0f, 0.1f);
			ImGui::Checkbox("Stable fit", &m_stableFit);
		}

		if (ImGui::TreeNodeEx("Strands settings", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::DragFloat("Curve subdivision factor", &m_renderInfoBlock.m_strandsSubdivisionXFactor, 1.0f, 1.0f, 1000.0f);
			ImGui::DragFloat("Ring subdivision factor", &m_renderInfoBlock.m_strandsSubdivisionYFactor, 1.0f, 1.0f, 1000.0f);
			ImGui::DragInt("Max curve subdivision", &m_renderInfoBlock.m_strandsSubdivisionMaxX, 1, 1, 15);
			ImGui::DragInt("Max ring subdivision", &m_renderInfoBlock.m_strandsSubdivisionMaxY, 1, 1, 15);

			ImGui::TreePop();
		}
		ImGui::End();
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

Handle RenderLayer::GetInstanceHandle(uint32_t index)
{
	const auto search = m_instanceHandles.find(index);
	if (search == m_instanceHandles.end())
	{
		return 0;
	}
	return search->second;
}


uint32_t RenderLayer::RegisterCameraIndex(const Handle& handle, const CameraInfoBlock& cameraInfoBlock)
{
	const auto search = m_cameraIndices.find(handle);
	if (search == m_cameraIndices.end())
	{
		const uint32_t index = m_cameraInfoBlocks.size();
		m_cameraIndices[handle] = index;
		m_cameraInfoBlocks.emplace_back(cameraInfoBlock);
		return index;
	}
	return search->second;
}

uint32_t RenderLayer::RegisterMaterialIndex(const Handle& handle, const MaterialInfoBlock& materialInfoBlock)
{
	const auto search = m_materialIndices.find(handle);
	if (search == m_materialIndices.end())
	{
		const uint32_t index = m_materialInfoBlocks.size();
		m_materialIndices[handle] = index;
		m_materialInfoBlocks.emplace_back(materialInfoBlock);
		return index;
	}
	return search->second;
}

uint32_t RenderLayer::RegisterInstanceIndex(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock)
{
	const auto search = m_instanceIndices.find(handle);
	if (search == m_instanceIndices.end())
	{
		const uint32_t index = m_instanceInfoBlocks.size();
		m_instanceIndices[handle] = index;
		m_instanceHandles[index] = handle;
		m_instanceInfoBlocks.emplace_back(instanceInfoBlock);
		return index;
	}
	return search->second;
}



void RenderLayer::CollectCameras(std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras)
{
	auto scene = GetScene();
	std::vector<std::pair<std::shared_ptr<Camera>, glm::vec3>> cameraPairs;

	if (auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		for (const auto& [cameraHandle, editorCamera] : editorLayer->m_editorCameras) {
			if (editorCamera.m_camera || editorCamera.m_camera->IsEnabled())
			{
				cameraPairs.emplace_back(editorCamera.m_camera, editorCamera.m_position);
				CameraInfoBlock cameraInfoBlock;
				GlobalTransform sceneCameraGT;
				sceneCameraGT.SetValue(editorCamera.m_position, editorCamera.m_rotation, glm::vec3(1.0f));
				editorCamera.m_camera->UpdateCameraInfoBlock(cameraInfoBlock, sceneCameraGT);
				RegisterCameraIndex(cameraHandle, cameraInfoBlock);

				cameras.emplace_back(sceneCameraGT, editorCamera.m_camera);
			}
		}
	}
	if (const std::vector<Entity>* cameraEntities = scene->UnsafeGetPrivateComponentOwnersList<Camera>())
	{
		for (const auto& i : *cameraEntities)
		{
			if (!scene->IsEntityEnabled(i)) continue;
			assert(scene->HasPrivateComponent<Camera>(i));
			auto camera = scene->GetOrSetPrivateComponent<Camera>(i).lock();
			if (!camera || !camera->IsEnabled()) continue;
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
		m_directionalLightInfoBlocks.resize(Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE * cameras.size());
		for (const auto& lightEntity : *directionalLightEntities)
		{
			if (!scene->IsEntityEnabled(lightEntity))
				continue;
			const auto dlc = scene->GetOrSetPrivateComponent<DirectionalLight>(lightEntity).lock();
			if (!dlc->IsEnabled())
				continue;
			m_renderInfoBlock.m_directionalLightSize++;
		}
		std::vector<glm::uvec3> viewPortResults;
		Lighting::AllocateAtlas(m_renderInfoBlock.m_directionalLightSize, Graphics::Settings::DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION, viewPortResults);
		for (const auto& [cameraGlobalTransform, camera] : cameras) {
			auto cameraIndex = GetCameraIndex(camera->GetHandle());
			for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
			{
				const auto blockIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i;
				auto& viewPort = m_directionalLightInfoBlocks[blockIndex].m_viewPort;
				viewPort.x = viewPortResults[i].x;
				viewPort.y = viewPortResults[i].y;
				viewPort.z = viewPortResults[i].z;
				viewPort.w = viewPortResults[i].z;
			}
		}

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
				const auto blockIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + directionalLightIndex;
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
#pragma region Fix Shimmering due to the movement of the camera
					glm::mat4 shadowMatrix = lightProjection * lightView;
					glm::vec4 shadowOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
					shadowOrigin = shadowMatrix * shadowOrigin;
					shadowOrigin = shadowOrigin * static_cast<float>(m_directionalLightInfoBlocks[blockIndex].m_viewPort.z) / 2.0f;
					glm::vec4 roundedOrigin = glm::round(shadowOrigin);
					glm::vec4 roundOffset = roundedOrigin - shadowOrigin;
					roundOffset = roundOffset * 2.0f / static_cast<float>(m_directionalLightInfoBlocks[blockIndex].m_viewPort.z);
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
		}

	}
}

void RenderLayer::CollectPointLights()
{
	const auto scene = GetScene();
	const auto mainCamera = scene->m_mainCamera.Get<Camera>();
	glm::vec3 mainCameraPosition = { 0, 0, 0 };
	if (mainCamera)
	{
		mainCameraPosition = scene->GetDataComponent<GlobalTransform>(mainCamera->GetOwner()).GetPosition();
	}
	const std::vector<Entity>* pointLightEntities =
		scene->UnsafeGetPrivateComponentOwnersList<PointLight>();
	m_renderInfoBlock.m_pointLightSize = 0;
	if (pointLightEntities && !pointLightEntities->empty())
	{
		m_pointLightInfoBlocks.resize(pointLightEntities->size());
		std::multimap<float, size_t> sortedPointLightIndices;
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

			sortedPointLightIndices.insert({ glm::distance(mainCameraPosition, position), m_renderInfoBlock.m_pointLightSize });
			m_renderInfoBlock.m_pointLightSize++;
		}
		std::vector<glm::uvec3> viewPortResults;
		Lighting::AllocateAtlas(m_renderInfoBlock.m_pointLightSize, Graphics::Settings::POINT_LIGHT_SHADOW_MAP_RESOLUTION, viewPortResults);
		int allocationIndex = 0;
		for (const auto& pointLightIndex : sortedPointLightIndices)
		{
			auto& viewPort = m_pointLightInfoBlocks[pointLightIndex.second].m_viewPort;
			viewPort.x = viewPortResults[allocationIndex].x;
			viewPort.y = viewPortResults[allocationIndex].y;
			viewPort.z = viewPortResults[allocationIndex].z;
			viewPort.w = viewPortResults[allocationIndex].z;

			allocationIndex++;
		}
	}
	m_pointLightInfoBlocks.resize(m_renderInfoBlock.m_pointLightSize);
}

void RenderLayer::CollectSpotLights()
{
	const auto scene = GetScene();
	const auto mainCamera = scene->m_mainCamera.Get<Camera>();
	glm::vec3 mainCameraPosition = { 0, 0, 0 };
	if (mainCamera)
	{
		mainCameraPosition = scene->GetDataComponent<GlobalTransform>(mainCamera->GetOwner()).GetPosition();
	}

	m_renderInfoBlock.m_spotLightSize = 0;
	const std::vector<Entity>* spotLightEntities =
		scene->UnsafeGetPrivateComponentOwnersList<SpotLight>();
	if (spotLightEntities && !spotLightEntities->empty())
	{
		m_spotLightInfoBlocks.resize(spotLightEntities->size());
		std::multimap<float, size_t> sortedSpotLightIndices;
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

			sortedSpotLightIndices.insert({ glm::distance(mainCameraPosition, position), m_renderInfoBlock.m_spotLightSize });
			m_renderInfoBlock.m_spotLightSize++;
		}
		std::vector<glm::uvec3> viewPortResults;
		Lighting::AllocateAtlas(m_renderInfoBlock.m_spotLightSize, Graphics::Settings::SPOT_LIGHT_SHADOW_MAP_RESOLUTION, viewPortResults);
		int allocationIndex = 0;
		for (const auto& spotLightIndex : sortedSpotLightIndices)
		{
			auto& viewPort = m_spotLightInfoBlocks[spotLightIndex.second].m_viewPort;
			viewPort.x = viewPortResults[allocationIndex].x;
			viewPort.y = viewPortResults[allocationIndex].y;
			viewPort.z = viewPortResults[allocationIndex].z;
			viewPort.w = viewPortResults[allocationIndex].z;
			allocationIndex++;
		}
	}
	m_spotLightInfoBlocks.resize(m_renderInfoBlock.m_spotLightSize);
}

void RenderLayer::ApplyAnimator() const
{
	const auto scene = GetScene();
	if (const auto* owners =
		scene->UnsafeGetPrivateComponentOwnersList<Animator>())
	{
		Jobs::ParallelFor(owners->size(), [&](unsigned i)
			{
				const auto entity = owners->at(i);
				if (!scene->IsEntityEnabled(entity)) return;
				const auto animator = scene->GetOrSetPrivateComponent<Animator>(owners->at(i)).lock();
				if (!animator->IsEnabled()) return;
				animator->Apply();
			});
	}
	if (const auto* owners =
		scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>())
	{
		Jobs::ParallelFor(owners->size(), [&](unsigned i)
			{
				const auto entity = owners->at(i);
				if (!scene->IsEntityEnabled(entity)) return;
				const auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(entity).lock();
				if (!skinnedMeshRenderer->IsEnabled()) return;
				skinnedMeshRenderer->UpdateBoneMatrices();
			}
		);
		for (const auto& i : *owners)
		{
			if (!scene->IsEntityEnabled(i)) return;
			const auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(i).lock();
			if (!skinnedMeshRenderer->IsEnabled()) return;
			skinnedMeshRenderer->UpdateBoneMatrices();
			skinnedMeshRenderer->m_boneMatrices->UploadData();
		}
	}
}

void RenderLayer::PreparePointAndSpotLightShadowMap() const
{
	const bool countShadowRenderingDrawCalls = m_countShadowRenderingDrawCalls;
	const bool useMeshShader = Graphics::Constants::ENABLE_MESH_SHADER && Graphics::Settings::USE_MESH_SHADER;
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	const auto& pointLightShadowPipeline = useMeshShader ? Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_MESH") : Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP");
	const auto& spotLightShadowPipeline = useMeshShader ? Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_MESH") : Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP");

	const auto& pointLightShadowSkinnedPipeline = Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_SKINNED");
	const auto& spotLightShadowSkinnedPipeline = Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_SKINNED");

	const auto& pointLightShadowInstancedPipeline = Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_INSTANCED");
	const auto& spotLightShadowInstancedPipeline = Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_INSTANCED");

	const auto& pointLightShadowStrandsPipeline = Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_STRANDS");
	const auto& spotLightShadowStrandsPipeline = Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_STRANDS");
	auto& graphics = Graphics::GetInstance();

	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
		{
#pragma region Viewport and scissor
			VkRect2D renderArea;
			renderArea.offset = { 0, 0 };
			renderArea.extent.width = m_lighting->m_pointLightShadowMap->GetExtent().width;
			renderArea.extent.height = m_lighting->m_pointLightShadowMap->GetExtent().height;

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


#pragma endregion
			m_lighting->m_pointLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);

			for (int face = 0; face < 6; face++) {
				GeometryStorage::BindVertices(commandBuffer);
				{
					VkRenderingInfo renderInfo{};
					auto depthAttachment = m_lighting->GetLayeredPointLightDepthAttachmentInfo(face, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					pointLightShadowPipeline->m_states.ResetAllStates(commandBuffer, 0);
					pointLightShadowPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
					pointLightShadowPipeline->m_states.m_viewPort = viewport;
					pointLightShadowPipeline->m_states.m_scissor = scissor;

					vkCmdBeginRendering(commandBuffer, &renderInfo);
					pointLightShadowPipeline->Bind(commandBuffer);
					pointLightShadowPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					for (int i = 0; i < m_pointLightInfoBlocks.size(); i++)
					{
						const auto& pointLightInfoBlock = m_pointLightInfoBlocks[i];
						viewport.x = pointLightInfoBlock.m_viewPort.x;
						viewport.y = pointLightInfoBlock.m_viewPort.y;
						viewport.width = pointLightInfoBlock.m_viewPort.z;
						viewport.height = pointLightInfoBlock.m_viewPort.w;
						pointLightShadowPipeline->m_states.m_viewPort = viewport;
						scissor.extent.width = viewport.width;
						scissor.extent.height = viewport.height;
						pointLightShadowPipeline->m_states.m_scissor = scissor;

						if (m_enableIndirectRendering) {
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = i;
							pushConstant.m_lightSplitIndex = face;
							pushConstant.m_instanceIndex = 0;
							pointLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
							pointLightShadowPipeline->m_states.ApplyAllStates(commandBuffer);
							if (useMeshShader) {
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
								vkCmdDrawMeshTasksIndirectEXT(commandBuffer, m_meshDrawMeshTasksIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawMeshTasksIndirectCommands.size(), sizeof(VkDrawMeshTasksIndirectCommandEXT));
							}
							else
							{
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
								vkCmdDrawIndexedIndirect(commandBuffer, m_meshDrawIndexedIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawIndexedIndirectCommands.size(), sizeof(VkDrawIndexedIndirectCommand));
							}
						}
						else {
							m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
								{
									if (!renderCommand.m_castShadow) return;
									RenderInstancePushConstant pushConstant;
									pushConstant.m_cameraIndex = i;
									pushConstant.m_lightSplitIndex = face;
									pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
									pointLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
									if (useMeshShader)
									{
										if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
										if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
										vkCmdDrawMeshTasksEXT(commandBuffer, 1, 1, 1);
									}
									else {
										const auto mesh = renderCommand.m_mesh;
										if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
										if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
										mesh->DrawIndexed(commandBuffer, pointLightShadowPipeline->m_states, 1);
									}
								}
							);
						}
					}
					vkCmdEndRendering(commandBuffer);
				}
				{
					VkRenderingInfo renderInfo{};
					auto depthAttachment = m_lighting->GetLayeredPointLightDepthAttachmentInfo(face, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					pointLightShadowInstancedPipeline->m_states.ResetAllStates(commandBuffer, 0);
					pointLightShadowInstancedPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
					pointLightShadowInstancedPipeline->m_states.m_viewPort = viewport;
					pointLightShadowInstancedPipeline->m_states.m_scissor = scissor;
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					pointLightShadowInstancedPipeline->Bind(commandBuffer);
					pointLightShadowInstancedPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					for (int i = 0; i < m_pointLightInfoBlocks.size(); i++)
					{
						const auto& pointLightInfoBlock = m_pointLightInfoBlocks[i];
						viewport.x = pointLightInfoBlock.m_viewPort.x;
						viewport.y = pointLightInfoBlock.m_viewPort.y;
						viewport.width = pointLightInfoBlock.m_viewPort.z;
						viewport.height = pointLightInfoBlock.m_viewPort.w;
						scissor.extent.width = viewport.width;
						scissor.extent.height = viewport.height;
						pointLightShadowInstancedPipeline->m_states.m_viewPort = viewport;
						pointLightShadowInstancedPipeline->m_states.m_scissor = scissor;
						m_deferredInstancedRenderInstances.Dispatch([&](const InstancedRenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								renderCommand.m_particleInfos->UploadData();
								pointLightShadowInstancedPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_particleInfos->GetDescriptorSet()->GetVkDescriptorSet());
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = i;
								pushConstant.m_lightSplitIndex = face;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								pointLightShadowInstancedPipeline->PushConstant(commandBuffer, 0, pushConstant);
								const auto mesh = renderCommand.m_mesh;
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size() * renderCommand.m_particleInfos->m_particleInfos.size();
								mesh->DrawIndexed(commandBuffer, pointLightShadowInstancedPipeline->m_states, renderCommand.m_particleInfos->m_particleInfos.size());
							}
						);
					}
					vkCmdEndRendering(commandBuffer);
				}
				GeometryStorage::BindSkinnedVertices(commandBuffer);
				{
					VkRenderingInfo renderInfo{};
					auto depthAttachment = m_lighting->GetLayeredPointLightDepthAttachmentInfo(face, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					pointLightShadowSkinnedPipeline->m_states.ResetAllStates(commandBuffer, 0);
					pointLightShadowSkinnedPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
					pointLightShadowSkinnedPipeline->m_states.m_viewPort = viewport;
					pointLightShadowSkinnedPipeline->m_states.m_scissor = scissor;
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					pointLightShadowSkinnedPipeline->Bind(commandBuffer);
					pointLightShadowSkinnedPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					for (int i = 0; i < m_pointLightInfoBlocks.size(); i++)
					{
						const auto& pointLightInfoBlock = m_pointLightInfoBlocks[i];
						viewport.x = pointLightInfoBlock.m_viewPort.x;
						viewport.y = pointLightInfoBlock.m_viewPort.y;
						viewport.width = pointLightInfoBlock.m_viewPort.z;
						viewport.height = pointLightInfoBlock.m_viewPort.w;
						scissor.extent.width = viewport.width;
						scissor.extent.height = viewport.height;
						pointLightShadowSkinnedPipeline->m_states.m_viewPort = viewport;
						pointLightShadowSkinnedPipeline->m_states.m_scissor = scissor;
						m_deferredSkinnedRenderInstances.Dispatch([&](const SkinnedRenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								pointLightShadowSkinnedPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->GetDescriptorSet()->GetVkDescriptorSet());
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = i;
								pushConstant.m_lightSplitIndex = face;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								pointLightShadowSkinnedPipeline->PushConstant(commandBuffer, 0, pushConstant);
								const auto skinnedMesh = renderCommand.m_skinnedMesh;
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_skinnedMesh->m_skinnedTriangles.size();
								skinnedMesh->DrawIndexed(commandBuffer, pointLightShadowSkinnedPipeline->m_states, 1);
							}
						);
					}
					vkCmdEndRendering(commandBuffer);
				}
				GeometryStorage::BindStrandPoints(commandBuffer);
				{
					VkRenderingInfo renderInfo{};
					auto depthAttachment = m_lighting->GetLayeredPointLightDepthAttachmentInfo(face, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					pointLightShadowStrandsPipeline->m_states.ResetAllStates(commandBuffer, 0);
					pointLightShadowStrandsPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
					pointLightShadowStrandsPipeline->m_states.m_viewPort = viewport;
					pointLightShadowStrandsPipeline->m_states.m_scissor = scissor;
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					pointLightShadowStrandsPipeline->Bind(commandBuffer);
					pointLightShadowStrandsPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					for (int i = 0; i < m_pointLightInfoBlocks.size(); i++)
					{
						const auto& pointLightInfoBlock = m_pointLightInfoBlocks[i];
						viewport.x = pointLightInfoBlock.m_viewPort.x;
						viewport.y = pointLightInfoBlock.m_viewPort.y;
						viewport.width = pointLightInfoBlock.m_viewPort.z;
						viewport.height = pointLightInfoBlock.m_viewPort.w;
						scissor.extent.width = viewport.width;
						scissor.extent.height = viewport.height;
						pointLightShadowStrandsPipeline->m_states.m_viewPort = viewport;
						pointLightShadowStrandsPipeline->m_states.m_scissor = scissor;
						m_deferredStrandsRenderInstances.Dispatch([&](const StrandsRenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = i;
								pushConstant.m_lightSplitIndex = face;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								pointLightShadowStrandsPipeline->PushConstant(commandBuffer, 0, pushConstant);
								const auto strands = renderCommand.m_strands;
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_strandsSegments[currentFrameIndex] += renderCommand.m_strands->m_segments.size();
								strands->DrawIndexed(commandBuffer, pointLightShadowStrandsPipeline->m_states, 1);
							}
						);
					}
					vkCmdEndRendering(commandBuffer);
				}
			}
#pragma region Viewport and scissor

			renderArea.offset = { 0, 0 };
			renderArea.extent.width = m_lighting->m_spotLightShadowMap->GetExtent().width;
			renderArea.extent.height = m_lighting->m_spotLightShadowMap->GetExtent().height;

			viewport.x = 0;
			viewport.y = 0;
			viewport.width = renderArea.extent.width;
			viewport.height = renderArea.extent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			scissor.offset = { 0, 0 };
			scissor.extent.width = renderArea.extent.width;
			scissor.extent.height = renderArea.extent.height;


#pragma endregion
			m_lighting->m_spotLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
			GeometryStorage::BindVertices(commandBuffer);
			{
				VkRenderingInfo renderInfo{};
				auto depthAttachment = m_lighting->GetSpotLightDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.colorAttachmentCount = 0;
				renderInfo.pColorAttachments = nullptr;
				renderInfo.pDepthAttachment = &depthAttachment;
				spotLightShadowPipeline->m_states.ResetAllStates(commandBuffer, 0);
				spotLightShadowPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
				spotLightShadowPipeline->m_states.m_viewPort = viewport;
				spotLightShadowPipeline->m_states.m_scissor = scissor;
				vkCmdBeginRendering(commandBuffer, &renderInfo);
				spotLightShadowPipeline->Bind(commandBuffer);
				spotLightShadowPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
				for (int i = 0; i < m_spotLightInfoBlocks.size(); i++)
				{
					const auto& spotLightInfoBlock = m_spotLightInfoBlocks[i];
					viewport.x = spotLightInfoBlock.m_viewPort.x;
					viewport.y = spotLightInfoBlock.m_viewPort.y;
					viewport.width = spotLightInfoBlock.m_viewPort.z;
					viewport.height = spotLightInfoBlock.m_viewPort.w;
					spotLightShadowPipeline->m_states.m_viewPort = viewport;
					scissor.extent.width = viewport.width;
					scissor.extent.height = viewport.height;
					spotLightShadowPipeline->m_states.m_scissor = scissor;
					if (m_enableIndirectRendering) {
						RenderInstancePushConstant pushConstant;
						pushConstant.m_cameraIndex = i;
						pushConstant.m_lightSplitIndex = 0;
						pushConstant.m_instanceIndex = 0;
						spotLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
						spotLightShadowPipeline->m_states.ApplyAllStates(commandBuffer);
						if (useMeshShader) {
							if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
							if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
							vkCmdDrawMeshTasksIndirectEXT(commandBuffer, m_meshDrawMeshTasksIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawMeshTasksIndirectCommands.size(), sizeof(VkDrawMeshTasksIndirectCommandEXT));
						}
						else
						{
							if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
							if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
							vkCmdDrawIndexedIndirect(commandBuffer, m_meshDrawIndexedIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawIndexedIndirectCommands.size(), sizeof(VkDrawIndexedIndirectCommand));
						}
					}
					else {
						m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = i;
								pushConstant.m_lightSplitIndex = 0;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								spotLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
								if (useMeshShader)
								{
									if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
									if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
									vkCmdDrawMeshTasksEXT(commandBuffer, 1, 1, 1);
								}
								else {
									const auto mesh = renderCommand.m_mesh;
									if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
									if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
									mesh->DrawIndexed(commandBuffer, spotLightShadowPipeline->m_states, 1);
								}
							}
						);
					}
				}
				vkCmdEndRendering(commandBuffer);
			}
			{
				VkRenderingInfo renderInfo{};
				auto depthAttachment = m_lighting->GetSpotLightDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.colorAttachmentCount = 0;
				renderInfo.pColorAttachments = nullptr;
				renderInfo.pDepthAttachment = &depthAttachment;
				spotLightShadowInstancedPipeline->m_states.ResetAllStates(commandBuffer, 0);
				spotLightShadowInstancedPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
				spotLightShadowInstancedPipeline->m_states.m_viewPort = viewport;
				spotLightShadowInstancedPipeline->m_states.m_scissor = scissor;
				vkCmdBeginRendering(commandBuffer, &renderInfo);
				spotLightShadowInstancedPipeline->Bind(commandBuffer);
				spotLightShadowInstancedPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
				for (int i = 0; i < m_spotLightInfoBlocks.size(); i++)
				{
					const auto& spotLightInfoBlock = m_spotLightInfoBlocks[i];
					viewport.x = spotLightInfoBlock.m_viewPort.x;
					viewport.y = spotLightInfoBlock.m_viewPort.y;
					viewport.width = spotLightInfoBlock.m_viewPort.z;
					viewport.height = spotLightInfoBlock.m_viewPort.w;
					spotLightShadowInstancedPipeline->m_states.m_viewPort = viewport;

					m_deferredInstancedRenderInstances.Dispatch([&](const InstancedRenderInstance& renderCommand)
						{
							if (!renderCommand.m_castShadow) return;
							renderCommand.m_particleInfos->UploadData();
							spotLightShadowInstancedPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_particleInfos->GetDescriptorSet()->GetVkDescriptorSet());
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = i;
							pushConstant.m_lightSplitIndex = 0;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							spotLightShadowInstancedPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto mesh = renderCommand.m_mesh;
							if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
							if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size() * renderCommand.m_particleInfos->m_particleInfos.size();
							mesh->DrawIndexed(commandBuffer, spotLightShadowInstancedPipeline->m_states, renderCommand.m_particleInfos->m_particleInfos.size());
						}
					);
				}
				vkCmdEndRendering(commandBuffer);
			}
			GeometryStorage::BindSkinnedVertices(commandBuffer);
			{
				VkRenderingInfo renderInfo{};
				auto depthAttachment = m_lighting->GetSpotLightDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.colorAttachmentCount = 0;
				renderInfo.pColorAttachments = nullptr;
				renderInfo.pDepthAttachment = &depthAttachment;
				spotLightShadowSkinnedPipeline->m_states.ResetAllStates(commandBuffer, 0);
				spotLightShadowSkinnedPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
				spotLightShadowSkinnedPipeline->m_states.m_viewPort = viewport;
				spotLightShadowSkinnedPipeline->m_states.m_scissor = scissor;
				vkCmdBeginRendering(commandBuffer, &renderInfo);
				spotLightShadowSkinnedPipeline->Bind(commandBuffer);
				spotLightShadowSkinnedPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
				for (int i = 0; i < m_spotLightInfoBlocks.size(); i++)
				{
					const auto& spotLightInfoBlock = m_spotLightInfoBlocks[i];
					viewport.x = spotLightInfoBlock.m_viewPort.x;
					viewport.y = spotLightInfoBlock.m_viewPort.y;
					viewport.width = spotLightInfoBlock.m_viewPort.z;
					viewport.height = spotLightInfoBlock.m_viewPort.w;
					spotLightShadowSkinnedPipeline->m_states.m_viewPort = viewport;

					m_deferredSkinnedRenderInstances.Dispatch([&](const SkinnedRenderInstance& renderCommand)
						{
							if (!renderCommand.m_castShadow) return;
							spotLightShadowSkinnedPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->GetDescriptorSet()->GetVkDescriptorSet());
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = i;
							pushConstant.m_lightSplitIndex = 0;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							spotLightShadowSkinnedPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto skinnedMesh = renderCommand.m_skinnedMesh;
							if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
							if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_skinnedMesh->m_skinnedTriangles.size();
							skinnedMesh->DrawIndexed(commandBuffer, spotLightShadowSkinnedPipeline->m_states, 1);
						}
					);
				}
				vkCmdEndRendering(commandBuffer);
			}
			GeometryStorage::BindStrandPoints(commandBuffer);
			{
				VkRenderingInfo renderInfo{};
				auto depthAttachment = m_lighting->GetSpotLightDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.colorAttachmentCount = 0;
				renderInfo.pColorAttachments = nullptr;
				renderInfo.pDepthAttachment = &depthAttachment;
				spotLightShadowStrandsPipeline->m_states.ResetAllStates(commandBuffer, 0);
				spotLightShadowStrandsPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
				spotLightShadowStrandsPipeline->m_states.m_viewPort = viewport;
				spotLightShadowStrandsPipeline->m_states.m_scissor = scissor;
				vkCmdBeginRendering(commandBuffer, &renderInfo);
				spotLightShadowStrandsPipeline->Bind(commandBuffer);
				spotLightShadowStrandsPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
				for (int i = 0; i < m_spotLightInfoBlocks.size(); i++)
				{
					const auto& spotLightInfoBlock = m_spotLightInfoBlocks[i];
					viewport.x = spotLightInfoBlock.m_viewPort.x;
					viewport.y = spotLightInfoBlock.m_viewPort.y;
					viewport.width = spotLightInfoBlock.m_viewPort.z;
					viewport.height = spotLightInfoBlock.m_viewPort.w;
					spotLightShadowStrandsPipeline->m_states.m_viewPort = viewport;

					m_deferredStrandsRenderInstances.Dispatch([&](const StrandsRenderInstance& renderCommand)
						{
							if (!renderCommand.m_castShadow) return;
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = i;
							pushConstant.m_lightSplitIndex = 0;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							spotLightShadowStrandsPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto strands = renderCommand.m_strands;
							if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
							if(countShadowRenderingDrawCalls) graphics.m_strandsSegments[currentFrameIndex] += renderCommand.m_strands->m_segments.size();
							strands->DrawIndexed(commandBuffer, spotLightShadowStrandsPipeline->m_states, 1);
						}
					);
				}
				vkCmdEndRendering(commandBuffer);
			}
			m_lighting->m_pointLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			m_lighting->m_spotLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		});
}

void RenderLayer::CollectRenderInstances(Bound& worldBound)
{
	m_needFade = false;
	auto scene = GetScene();
	auto editorLayer = Application::GetLayer<EditorLayer>();
	const bool enableSelectionHighLight = editorLayer && scene->IsEntityValid(editorLayer->m_selectedEntity);
	auto& minBound = worldBound.m_min;
	auto& maxBound = worldBound.m_max;
	minBound = glm::vec3(FLT_MAX);
	maxBound = glm::vec3(FLT_MIN);

	if (const auto* owners =
		scene->UnsafeGetPrivateComponentOwnersList<MeshRenderer>())
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(owner).lock();
			auto material = meshRenderer->m_material.Get<Material>();
			auto mesh = meshRenderer->m_mesh.Get<Mesh>();
			if (!meshRenderer->IsEnabled() || !material || !mesh || !mesh->m_meshletRange || !mesh->m_triangleRange)
				continue;
			if (mesh->UnsafeGetVertices().empty() || mesh->UnsafeGetTriangles().empty()) continue;
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
			instanceInfoBlock.m_materialIndex = materialIndex;
			instanceInfoBlock.m_entitySelected = enableSelectionHighLight && scene->IsEntityAncestorSelected(owner) ? 1 : 0;

			instanceInfoBlock.m_meshletIndexOffset = mesh->m_meshletRange->m_offset;
			instanceInfoBlock.m_meshletSize = mesh->m_meshletRange->m_size;

			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);
			RenderInstance renderInstance;
			renderInstance.m_commandType = RenderCommandType::FromRenderer;
			renderInstance.m_owner = owner;
			renderInstance.m_mesh = mesh;
			renderInstance.m_castShadow = meshRenderer->m_castShadow;
			renderInstance.m_meshletSize = mesh->m_meshletRange->m_size;
			renderInstance.m_instanceIndex = instanceIndex;

			renderInstance.m_lineWidth = material->m_drawSettings.m_lineWidth;
			renderInstance.m_cullMode = material->m_drawSettings.m_cullMode;
			renderInstance.m_polygonMode = material->m_drawSettings.m_polygonMode;
			if (instanceInfoBlock.m_entitySelected == 1) m_needFade = true;
			if (material->m_drawSettings.m_blending)
			{
				m_transparentRenderInstances.m_renderCommands.push_back(renderInstance);
			}
			else
			{
				m_deferredRenderInstances.m_renderCommands.push_back(renderInstance);
			}

			auto& newMeshTask = m_meshDrawMeshTasksIndirectCommands.emplace_back();
			newMeshTask.groupCountX = 1;
			newMeshTask.groupCountY = 1;
			newMeshTask.groupCountZ = 1;

			auto& newDrawTask = m_meshDrawIndexedIndirectCommands.emplace_back();
			newDrawTask.instanceCount = 1;
			newDrawTask.firstIndex = mesh->m_triangleRange->m_offset * 3;
			newDrawTask.indexCount = static_cast<uint32_t>(mesh->m_triangles.size() * 3);
			newDrawTask.vertexOffset = 0;
			newDrawTask.firstInstance = 0;

			m_totalMeshTriangles += mesh->m_triangles.size();
		}
	}
	if (const auto* owners = scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>())
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(owner).lock();
			auto material = skinnedMeshRenderer->m_material.Get<Material>();
			auto skinnedMesh = skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>();
			if (!skinnedMeshRenderer->IsEnabled() || !material || !skinnedMesh || !skinnedMesh->m_skinnedMeshletRange || !skinnedMesh->m_skinnedTriangleRange)
				continue;
			if (skinnedMesh->m_skinnedVertices.empty() || skinnedMesh->m_skinnedTriangles.empty()) continue;
			GlobalTransform gt;
			if (auto animator = skinnedMeshRenderer->m_animator.Get<Animator>(); !animator)
			{
				continue;
			}
			if (!skinnedMeshRenderer->m_ragDoll)
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

			MaterialInfoBlock materialInfoBlock;
			material->UpdateMaterialInfoBlock(materialInfoBlock);
			auto materialIndex = RegisterMaterialIndex(material->GetHandle(), materialInfoBlock);
			InstanceInfoBlock instanceInfoBlock;
			instanceInfoBlock.m_model = gt;
			instanceInfoBlock.m_materialIndex = materialIndex;
			instanceInfoBlock.m_entitySelected = enableSelectionHighLight && scene->IsEntityAncestorSelected(owner) ? 1 : 0;
			instanceInfoBlock.m_meshletSize = skinnedMesh->m_skinnedMeshletRange->m_size;
			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);

			SkinnedRenderInstance renderInstance;
			renderInstance.m_commandType = RenderCommandType::FromRenderer;
			renderInstance.m_owner = owner;
			renderInstance.m_skinnedMesh = skinnedMesh;
			renderInstance.m_castShadow = skinnedMeshRenderer->m_castShadow;
			renderInstance.m_boneMatrices = skinnedMeshRenderer->m_boneMatrices;
			renderInstance.m_skinnedMeshletSize = skinnedMesh->m_skinnedMeshletRange->m_size;
			renderInstance.m_instanceIndex = instanceIndex;

			renderInstance.m_lineWidth = material->m_drawSettings.m_lineWidth;
			renderInstance.m_cullMode = material->m_drawSettings.m_cullMode;
			renderInstance.m_polygonMode = material->m_drawSettings.m_polygonMode;
			if (instanceInfoBlock.m_entitySelected == 1) m_needFade = true;

			if (material->m_drawSettings.m_blending)
			{
				m_transparentSkinnedRenderInstances.m_renderCommands.push_back(renderInstance);
			}
			else
			{
				m_deferredSkinnedRenderInstances.m_renderCommands.push_back(renderInstance);
			}
		}
	}

	if (const auto* owners = scene->UnsafeGetPrivateComponentOwnersList<Particles>())
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto particles = scene->GetOrSetPrivateComponent<Particles>(owner).lock();
			auto material = particles->m_material.Get<Material>();
			auto mesh = particles->m_mesh.Get<Mesh>();
			auto particleInfoList = particles->m_particleInfoList.Get<ParticleInfoList>();
			if (!particles->IsEnabled() || !material || !mesh || !mesh->m_meshletRange || !mesh->m_triangleRange || !particleInfoList)
				continue;
			if (particleInfoList->m_particleInfos.empty()) continue;
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
			instanceInfoBlock.m_materialIndex = materialIndex;
			instanceInfoBlock.m_entitySelected = enableSelectionHighLight && scene->IsEntityAncestorSelected(owner) ? 1 : 0;
			instanceInfoBlock.m_meshletSize = mesh->m_meshletRange->m_size;
			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);

			InstancedRenderInstance renderInstance;
			renderInstance.m_commandType = RenderCommandType::FromRenderer;
			renderInstance.m_owner = owner;
			renderInstance.m_mesh = mesh;
			renderInstance.m_castShadow = particles->m_castShadow;
			renderInstance.m_particleInfos = particleInfoList;
			renderInstance.m_meshletSize = mesh->m_meshletRange->m_size;

			renderInstance.m_instanceIndex = instanceIndex;

			renderInstance.m_lineWidth = material->m_drawSettings.m_lineWidth;
			renderInstance.m_cullMode = material->m_drawSettings.m_cullMode;
			renderInstance.m_polygonMode = material->m_drawSettings.m_polygonMode;

			if (instanceInfoBlock.m_entitySelected == 1) m_needFade = true;

			if (material->m_drawSettings.m_blending)
			{
				m_transparentInstancedRenderInstances.m_renderCommands.push_back(renderInstance);
			}
			else
			{
				m_deferredInstancedRenderInstances.m_renderCommands.push_back(renderInstance);
			}
		}
	}

	if (const auto* owners = scene->UnsafeGetPrivateComponentOwnersList<StrandsRenderer>())
	{
		for (auto owner : *owners)
		{
			if (!scene->IsEntityEnabled(owner))
				continue;
			auto strandsRenderer = scene->GetOrSetPrivateComponent<StrandsRenderer>(owner).lock();
			auto material = strandsRenderer->m_material.Get<Material>();
			auto strands = strandsRenderer->m_strands.Get<Strands>();
			if (!strandsRenderer->IsEnabled() || !material || !strands || !strands->m_strandMeshletRange || !strands->m_segmentRange)
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

			MaterialInfoBlock materialInfoBlock;
			material->UpdateMaterialInfoBlock(materialInfoBlock);
			auto materialIndex = RegisterMaterialIndex(material->GetHandle(), materialInfoBlock);
			InstanceInfoBlock instanceInfoBlock;
			instanceInfoBlock.m_model = gt;
			instanceInfoBlock.m_materialIndex = materialIndex;
			instanceInfoBlock.m_entitySelected = enableSelectionHighLight && scene->IsEntityAncestorSelected(owner) ? 1 : 0;
			instanceInfoBlock.m_meshletSize = strands->m_strandMeshletRange->m_size;
			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);

			StrandsRenderInstance renderInstance;
			renderInstance.m_commandType = RenderCommandType::FromRenderer;
			renderInstance.m_owner = owner;
			renderInstance.m_strands = strands;
			renderInstance.m_castShadow = strandsRenderer->m_castShadow;
			renderInstance.m_strandMeshletSize = strands->m_strandMeshletRange->m_size;

			renderInstance.m_instanceIndex = instanceIndex;

			renderInstance.m_lineWidth = material->m_drawSettings.m_lineWidth;
			renderInstance.m_cullMode = material->m_drawSettings.m_cullMode;
			renderInstance.m_polygonMode = material->m_drawSettings.m_polygonMode;

			if (instanceInfoBlock.m_entitySelected == 1) m_needFade = true;

			if (material->m_drawSettings.m_blending)
			{
				m_transparentStrandsRenderInstances.m_renderCommands.push_back(renderInstance);
			}
			else
			{
				m_deferredStrandsRenderInstances.m_renderCommands.push_back(renderInstance);
			}
		}
	}
}




void RenderLayer::CreateStandardDescriptorBuffers()
{
#pragma region Standard Descrioptor Layout
	m_renderInfoDescriptorBuffers.clear();
	m_environmentInfoDescriptorBuffers.clear();
	m_cameraInfoDescriptorBuffers.clear();
	m_materialInfoDescriptorBuffers.clear();
	m_instanceInfoDescriptorBuffers.clear();

	m_kernelDescriptorBuffers.clear();
	m_directionalLightInfoDescriptorBuffers.clear();
	m_pointLightInfoDescriptorBuffers.clear();
	m_spotLightInfoDescriptorBuffers.clear();


	m_meshDrawMeshTasksIndirectCommandsBuffers.clear();
	m_meshDrawIndexedIndirectCommandsBuffers.clear();

	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo bufferVmaAllocationCreateInfo{};
	bufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	for (size_t i = 0; i < maxFrameInFlight; i++) {
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(RenderInfoBlock);
		m_renderInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(EnvironmentInfoBlock);
		m_environmentInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(CameraInfoBlock) * Graphics::Constants::INITIAL_CAMERA_SIZE;
		m_cameraInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(MaterialInfoBlock) * Graphics::Constants::INITIAL_MATERIAL_SIZE;
		m_materialInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(InstanceInfoBlock) * Graphics::Constants::INITIAL_INSTANCE_SIZE;
		m_instanceInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));


		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(glm::vec4) * Graphics::Constants::MAX_KERNEL_AMOUNT * 2;
		m_kernelDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(DirectionalLightInfo) * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE * Graphics::Constants::INITIAL_CAMERA_SIZE;
		m_directionalLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(PointLightInfo) * Graphics::Settings::MAX_POINT_LIGHT_SIZE;
		m_pointLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(SpotLightInfo) * Graphics::Settings::MAX_SPOT_LIGHT_SIZE;
		m_spotLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));

		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
		bufferCreateInfo.size = sizeof(VkDrawMeshTasksIndirectCommandEXT) * Graphics::Constants::INITIAL_RENDER_TASK_SIZE;
		m_meshDrawMeshTasksIndirectCommandsBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(VkDrawIndexedIndirectCommand) * Graphics::Constants::INITIAL_RENDER_TASK_SIZE;
		m_meshDrawIndexedIndirectCommandsBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
	}
#pragma endregion
}



void RenderLayer::CreatePerFrameDescriptorSets()
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();

	m_perFrameDescriptorSets.clear();
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		auto descriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("PER_FRAME_LAYOUT"));
		m_perFrameDescriptorSets.emplace_back(descriptorSet);
	}
}

void RenderLayer::PrepareEnvironmentalBrdfLut()
{
	m_environmentalBRDFLut.reset();
	m_environmentalBRDFLut = ProjectManager::CreateTemporaryAsset<Texture2D>();
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

		m_environmentalBRDFLut->m_image = std::make_unique<Image>(imageInfo);

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

		m_environmentalBRDFLut->m_imageView = std::make_unique<ImageView>(viewInfo);

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

		m_environmentalBRDFLut->m_sampler = std::make_unique<Sampler>(samplerInfo);
	}
	TextureStorage::RegisterTexture2D(m_environmentalBRDFLut);
	auto environmentalBrdfPipeline = Graphics::GetGraphicsPipeline("ENVIRONMENTAL_MAP_BRDF");
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
		m_environmentalBRDFLut->m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
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
		environmentalBrdfPipeline->m_states.m_viewPort = viewport;
		environmentalBrdfPipeline->m_states.m_scissor = scissor;
#pragma endregion
#pragma region Lighting pass
		{
			VkRenderingAttachmentInfo attachment{};
			attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

			attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			attachment.clearValue = { 0, 0, 0, 1 };
			attachment.imageView = m_environmentalBRDFLut->m_imageView->GetVkImageView();

			//const auto depthAttachment = camera->GetRenderTexture()->GetDepthAttachmentInfo();
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 1;
			renderInfo.pColorAttachments = &attachment;
			//renderInfo.pDepthAttachment = &depthAttachment;
			environmentalBrdfPipeline->m_states.m_depthTest = false;
			environmentalBrdfPipeline->m_states.m_colorBlendAttachmentStates.clear();
			environmentalBrdfPipeline->m_states.m_colorBlendAttachmentStates.resize(1);
			for (auto& i : environmentalBrdfPipeline->m_states.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
				i.blendEnable = VK_FALSE;
			}
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			environmentalBrdfPipeline->Bind(commandBuffer);
			const auto mesh = Resources::GetResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");
			GeometryStorage::BindVertices(commandBuffer);
			mesh->DrawIndexed(commandBuffer, environmentalBrdfPipeline->m_states, 1);
			vkCmdEndRendering(commandBuffer);
#pragma endregion
		}
		m_environmentalBRDFLut->m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	);
}




void RenderLayer::RenderToCamera(const GlobalTransform& cameraGlobalTransform, const std::shared_ptr<Camera>& camera)
{
	const bool countShadowRenderingDrawCalls = m_countShadowRenderingDrawCalls;
	const bool useMeshShader = Graphics::Constants::ENABLE_MESH_SHADER && Graphics::Settings::USE_MESH_SHADER;
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	const int cameraIndex = GetCameraIndex(camera->GetHandle());
	const auto scene = Application::GetActiveScene();
#pragma region Directional Light Shadows
	const auto& directionalLightShadowPipeline = useMeshShader ? Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_MESH")
		: Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP");
	const auto& directionalLightShadowPipelineSkinned = Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED");
	const auto& directionalLightShadowPipelineInstanced = Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_INSTANCED");
	const auto& directionalLightShadowPipelineStrands = Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS");
	auto& graphics = Graphics::GetInstance();
	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
		{
#pragma region Viewport and scissor
			VkRect2D renderArea;
			renderArea.offset = { 0, 0 };
			renderArea.extent.width = m_lighting->m_directionalLightShadowMap->GetExtent().width;
			renderArea.extent.height = m_lighting->m_directionalLightShadowMap->GetExtent().height;

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


#pragma endregion
			m_lighting->m_directionalLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);

			for (int split = 0; split < 4; split++) {

				{
					const auto depthAttachment = m_lighting->GetLayeredDirectionalLightDepthAttachmentInfo(split, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
					VkRenderingInfo renderInfo{};
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					directionalLightShadowPipeline->m_states.m_scissor = scissor;
					directionalLightShadowPipeline->m_states.m_viewPort = viewport;
					directionalLightShadowPipeline->m_states.m_colorBlendAttachmentStates.clear();

					vkCmdBeginRendering(commandBuffer, &renderInfo);
					directionalLightShadowPipeline->Bind(commandBuffer);
					directionalLightShadowPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					directionalLightShadowPipeline->m_states.m_cullMode = VK_CULL_MODE_NONE;
					GeometryStorage::BindVertices(commandBuffer);
					for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
					{
						const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i];
						viewport.x = directionalLightInfoBlock.m_viewPort.x;
						viewport.y = directionalLightInfoBlock.m_viewPort.y;
						viewport.width = directionalLightInfoBlock.m_viewPort.z;
						viewport.height = directionalLightInfoBlock.m_viewPort.w;
						scissor.extent.width = directionalLightInfoBlock.m_viewPort.z;
						scissor.extent.height = directionalLightInfoBlock.m_viewPort.w;
						directionalLightShadowPipeline->m_states.m_scissor = scissor;
						directionalLightShadowPipeline->m_states.m_viewPort = viewport;
						directionalLightShadowPipeline->m_states.ApplyAllStates(commandBuffer);
						if (m_enableIndirectRendering) {
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i;
							pushConstant.m_lightSplitIndex = split;
							pushConstant.m_instanceIndex = 0;
							directionalLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
							directionalLightShadowPipeline->m_states.ApplyAllStates(commandBuffer);
							if (useMeshShader) {
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
								vkCmdDrawMeshTasksIndirectEXT(commandBuffer, m_meshDrawMeshTasksIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawMeshTasksIndirectCommands.size(), sizeof(VkDrawMeshTasksIndirectCommandEXT));
							}
							else
							{
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
								vkCmdDrawIndexedIndirect(commandBuffer, m_meshDrawIndexedIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawIndexedIndirectCommands.size(), sizeof(VkDrawIndexedIndirectCommand));
							}
						}
						else {
							m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
								{
									if (!renderCommand.m_castShadow) return;
									RenderInstancePushConstant pushConstant;
									pushConstant.m_cameraIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i;
									pushConstant.m_lightSplitIndex = split;
									pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
									directionalLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
									if (useMeshShader)
									{
										if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
										if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
										vkCmdDrawMeshTasksEXT(commandBuffer, 1, 1, 1);
									}
									else {
										const auto mesh = renderCommand.m_mesh;
										GeometryStorage::BindVertices(commandBuffer);
										if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
										if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
										mesh->DrawIndexed(commandBuffer, directionalLightShadowPipeline->m_states, 1);
									}
								}
							);
						}

					}
					vkCmdEndRendering(commandBuffer);
				}
				{
					const auto depthAttachment = m_lighting->GetLayeredDirectionalLightDepthAttachmentInfo(split, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
					VkRenderingInfo renderInfo{};
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					directionalLightShadowPipelineInstanced->m_states.m_scissor = scissor;
					directionalLightShadowPipelineInstanced->m_states.m_viewPort = viewport;
					directionalLightShadowPipelineInstanced->m_states.m_colorBlendAttachmentStates.clear();

					vkCmdBeginRendering(commandBuffer, &renderInfo);
					directionalLightShadowPipelineInstanced->Bind(commandBuffer);
					directionalLightShadowPipelineInstanced->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					directionalLightShadowPipelineInstanced->m_states.m_cullMode = VK_CULL_MODE_NONE;
					GeometryStorage::BindVertices(commandBuffer);
					for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
					{
						const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i];
						viewport.x = directionalLightInfoBlock.m_viewPort.x;
						viewport.y = directionalLightInfoBlock.m_viewPort.y;
						viewport.width = directionalLightInfoBlock.m_viewPort.z;
						viewport.height = directionalLightInfoBlock.m_viewPort.w;
						scissor.extent.width = directionalLightInfoBlock.m_viewPort.z;
						scissor.extent.height = directionalLightInfoBlock.m_viewPort.w;
						directionalLightShadowPipelineInstanced->m_states.m_scissor = scissor;
						directionalLightShadowPipelineInstanced->m_states.m_viewPort = viewport;
						directionalLightShadowPipelineInstanced->m_states.ApplyAllStates(commandBuffer);

						m_deferredInstancedRenderInstances.Dispatch([&](const InstancedRenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								renderCommand.m_particleInfos->UploadData();
								directionalLightShadowPipelineInstanced->BindDescriptorSet(commandBuffer, 1, renderCommand.m_particleInfos->GetDescriptorSet()->GetVkDescriptorSet());
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i;
								pushConstant.m_lightSplitIndex = split;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								directionalLightShadowPipelineInstanced->PushConstant(commandBuffer, 0, pushConstant);
								const auto mesh = renderCommand.m_mesh;
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size() * renderCommand.m_particleInfos->m_particleInfos.size();
								mesh->DrawIndexed(commandBuffer, directionalLightShadowPipelineInstanced->m_states, renderCommand.m_particleInfos->m_particleInfos.size());
							}
						);

					}
					vkCmdEndRendering(commandBuffer);
				}
				GeometryStorage::BindSkinnedVertices(commandBuffer);
				{
					const auto depthAttachment = m_lighting->GetLayeredDirectionalLightDepthAttachmentInfo(split, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
					VkRenderingInfo renderInfo{};
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					directionalLightShadowPipelineSkinned->m_states.m_scissor = scissor;
					directionalLightShadowPipelineSkinned->m_states.m_viewPort = viewport;
					directionalLightShadowPipelineSkinned->m_states.m_colorBlendAttachmentStates.clear();

					vkCmdBeginRendering(commandBuffer, &renderInfo);
					directionalLightShadowPipelineSkinned->Bind(commandBuffer);
					directionalLightShadowPipelineSkinned->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					directionalLightShadowPipelineSkinned->m_states.m_cullMode = VK_CULL_MODE_NONE;
					for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
					{
						const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i];
						viewport.x = directionalLightInfoBlock.m_viewPort.x;
						viewport.y = directionalLightInfoBlock.m_viewPort.y;
						viewport.width = directionalLightInfoBlock.m_viewPort.z;
						viewport.height = directionalLightInfoBlock.m_viewPort.w;
						scissor.extent.width = directionalLightInfoBlock.m_viewPort.z;
						scissor.extent.height = directionalLightInfoBlock.m_viewPort.w;
						directionalLightShadowPipelineSkinned->m_states.m_scissor = scissor;
						directionalLightShadowPipelineSkinned->m_states.m_viewPort = viewport;
						directionalLightShadowPipelineSkinned->m_states.ApplyAllStates(commandBuffer);

						m_deferredSkinnedRenderInstances.Dispatch([&](const SkinnedRenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								directionalLightShadowPipelineSkinned->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->GetDescriptorSet()->GetVkDescriptorSet());
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i;
								pushConstant.m_lightSplitIndex = split;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								directionalLightShadowPipelineSkinned->PushConstant(commandBuffer, 0, pushConstant);
								const auto skinnedMesh = renderCommand.m_skinnedMesh;
								if (countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if (countShadowRenderingDrawCalls) graphics.m_triangles[currentFrameIndex] += renderCommand.m_skinnedMesh->m_skinnedTriangles.size();
								skinnedMesh->DrawIndexed(commandBuffer, directionalLightShadowPipelineSkinned->m_states, 1);
							}
						);

					}
					vkCmdEndRendering(commandBuffer);
				}
				GeometryStorage::BindStrandPoints(commandBuffer);
				{
					const auto depthAttachment = m_lighting->GetLayeredDirectionalLightDepthAttachmentInfo(split, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
					VkRenderingInfo renderInfo{};
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					directionalLightShadowPipelineStrands->m_states.m_scissor = scissor;
					directionalLightShadowPipelineStrands->m_states.m_viewPort = viewport;
					directionalLightShadowPipelineStrands->m_states.m_colorBlendAttachmentStates.clear();

					vkCmdBeginRendering(commandBuffer, &renderInfo);
					directionalLightShadowPipelineStrands->Bind(commandBuffer);
					directionalLightShadowPipelineStrands->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
					directionalLightShadowPipelineStrands->m_states.m_cullMode = VK_CULL_MODE_NONE;
					for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
					{
						const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i];
						viewport.x = directionalLightInfoBlock.m_viewPort.x;
						viewport.y = directionalLightInfoBlock.m_viewPort.y;
						viewport.width = directionalLightInfoBlock.m_viewPort.z;
						viewport.height = directionalLightInfoBlock.m_viewPort.w;
						scissor.extent.width = directionalLightInfoBlock.m_viewPort.z;
						scissor.extent.height = directionalLightInfoBlock.m_viewPort.w;
						directionalLightShadowPipelineStrands->m_states.m_scissor = scissor;
						directionalLightShadowPipelineStrands->m_states.m_viewPort = viewport;
						directionalLightShadowPipelineStrands->m_states.ApplyAllStates(commandBuffer);

						m_deferredStrandsRenderInstances.Dispatch([&](const StrandsRenderInstance& renderCommand)
							{
								if (!renderCommand.m_castShadow) return;
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = cameraIndex * Graphics::Settings::MAX_DIRECTIONAL_LIGHT_SIZE + i;
								pushConstant.m_lightSplitIndex = split;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								directionalLightShadowPipelineStrands->PushConstant(commandBuffer, 0, pushConstant);
								const auto strands = renderCommand.m_strands;
								if(countShadowRenderingDrawCalls) graphics.m_drawCall[currentFrameIndex]++;
								if(countShadowRenderingDrawCalls) graphics.m_strandsSegments[currentFrameIndex] += renderCommand.m_strands->m_segments.size();
								strands->DrawIndexed(commandBuffer, directionalLightShadowPipelineStrands->m_states, 1);
							}
						);

					}
					vkCmdEndRendering(commandBuffer);
				}
			}
		}
	);

#pragma endregion
	const auto editorLayer = Application::GetLayer<EditorLayer>();
	bool isSceneCamera = false;
	bool needFade = false;
	if (editorLayer)
	{
		if (camera.get() == editorLayer->GetSceneCamera().get()) isSceneCamera = true;
		if (m_needFade && editorLayer->m_highlightSelection) needFade = true;
	}

#pragma region Deferred Rendering
	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer) {
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

		camera->TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
		camera->m_renderTexture->GetDepthImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);

		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = renderArea;
		renderInfo.layerCount = 1;
#pragma endregion
#pragma region Geometry pass
		GeometryStorage::BindVertices(commandBuffer);
		{
			const auto depthAttachment = camera->m_renderTexture->GetDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.pDepthAttachment = &depthAttachment;
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();

			const auto& deferredPrepassPipeline = m_enableDebugVisualization ? useMeshShader ? Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_MESHLET_COLORED_PREPASS_MESH") : Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_MESHLET_COLORED_PREPASS") : useMeshShader ? Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_PREPASS_MESH") : Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_PREPASS");
			deferredPrepassPipeline->m_states.ResetAllStates(commandBuffer, colorAttachmentInfos.size());
			deferredPrepassPipeline->m_states.m_viewPort = viewport;
			deferredPrepassPipeline->m_states.m_scissor = scissor;
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredPrepassPipeline->Bind(commandBuffer);
			deferredPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());

			if (m_enableIndirectRendering) {
				if (!m_deferredRenderInstances.m_renderCommands.empty()) {
					RenderInstancePushConstant pushConstant;
					pushConstant.m_cameraIndex = cameraIndex;
					pushConstant.m_instanceIndex = 0;
					deferredPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
					if (useMeshShader) {
						graphics.m_drawCall[currentFrameIndex]++;
						graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
						vkCmdDrawMeshTasksIndirectEXT(commandBuffer, m_meshDrawMeshTasksIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawMeshTasksIndirectCommands.size(), sizeof(VkDrawMeshTasksIndirectCommandEXT));
					}
					else
					{
						GeometryStorage::BindVertices(commandBuffer);
						graphics.m_drawCall[currentFrameIndex]++;
						graphics.m_triangles[currentFrameIndex] += m_totalMeshTriangles;
						vkCmdDrawIndexedIndirect(commandBuffer, m_meshDrawIndexedIndirectCommandsBuffers[currentFrameIndex]->GetVkBuffer(), 0, m_meshDrawIndexedIndirectCommands.size(), sizeof(VkDrawIndexedIndirectCommand));
					}
				}
			}
			else {
				m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
					{
						RenderInstancePushConstant pushConstant;
						pushConstant.m_cameraIndex = cameraIndex;
						pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
						deferredPrepassPipeline->m_states.m_polygonMode = renderCommand.m_polygonMode;
						deferredPrepassPipeline->m_states.m_cullMode = renderCommand.m_cullMode;
						deferredPrepassPipeline->m_states.m_lineWidth = renderCommand.m_lineWidth;
						deferredPrepassPipeline->m_states.ApplyAllStates(commandBuffer);
						deferredPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
						if (useMeshShader)
						{
							graphics.m_drawCall[currentFrameIndex]++;
							graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
							vkCmdDrawMeshTasksEXT(commandBuffer, 1, 1, 1);
						}
						else {
							const auto mesh = renderCommand.m_mesh;
							graphics.m_drawCall[currentFrameIndex]++;
							graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size();
							mesh->DrawIndexed(commandBuffer, deferredPrepassPipeline->m_states, 1);
						}
					}
				);
			}

			vkCmdEndRendering(commandBuffer);
		}
		{
			const auto depthAttachment = camera->m_renderTexture->GetDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.pDepthAttachment = &depthAttachment;
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();

			const auto& deferredInstancedPrepassPipeline = Graphics::GetGraphicsPipeline("STANDARD_INSTANCED_DEFERRED_PREPASS");
			deferredInstancedPrepassPipeline->m_states.ResetAllStates(commandBuffer, colorAttachmentInfos.size());
			deferredInstancedPrepassPipeline->m_states.m_viewPort = viewport;
			deferredInstancedPrepassPipeline->m_states.m_scissor = scissor;
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredInstancedPrepassPipeline->Bind(commandBuffer);
			deferredInstancedPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			m_deferredInstancedRenderInstances.Dispatch([&](const InstancedRenderInstance& renderCommand)
				{
					renderCommand.m_particleInfos->UploadData();
					deferredInstancedPrepassPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_particleInfos->GetDescriptorSet()->GetVkDescriptorSet());
					RenderInstancePushConstant pushConstant;
					pushConstant.m_cameraIndex = cameraIndex;
					pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
					deferredInstancedPrepassPipeline->m_states.m_polygonMode = renderCommand.m_polygonMode;
					deferredInstancedPrepassPipeline->m_states.m_cullMode = renderCommand.m_cullMode;
					deferredInstancedPrepassPipeline->m_states.m_lineWidth = renderCommand.m_lineWidth;
					deferredInstancedPrepassPipeline->m_states.ApplyAllStates(commandBuffer);
					deferredInstancedPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
					const auto mesh = renderCommand.m_mesh;
					graphics.m_drawCall[currentFrameIndex]++;
					graphics.m_triangles[currentFrameIndex] += renderCommand.m_mesh->m_triangles.size() * renderCommand.m_particleInfos->m_particleInfos.size();
					mesh->DrawIndexed(commandBuffer, deferredInstancedPrepassPipeline->m_states, renderCommand.m_particleInfos->m_particleInfos.size());
				}
			);

			vkCmdEndRendering(commandBuffer);
		}
		GeometryStorage::BindSkinnedVertices(commandBuffer);
		{
			const auto depthAttachment = camera->m_renderTexture->GetDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.pDepthAttachment = &depthAttachment;
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();

			const auto& deferredSkinnedPrepassPipeline = Graphics::GetGraphicsPipeline("STANDARD_SKINNED_DEFERRED_PREPASS");
			deferredSkinnedPrepassPipeline->m_states.ResetAllStates(commandBuffer, colorAttachmentInfos.size());
			deferredSkinnedPrepassPipeline->m_states.m_viewPort = viewport;
			deferredSkinnedPrepassPipeline->m_states.m_scissor = scissor;
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredSkinnedPrepassPipeline->Bind(commandBuffer);
			deferredSkinnedPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			m_deferredSkinnedRenderInstances.Dispatch([&](const SkinnedRenderInstance& renderCommand)
				{
					deferredSkinnedPrepassPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->GetDescriptorSet()->GetVkDescriptorSet());
					RenderInstancePushConstant pushConstant;
					pushConstant.m_cameraIndex = cameraIndex;
					pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
					deferredSkinnedPrepassPipeline->m_states.m_polygonMode = renderCommand.m_polygonMode;
					deferredSkinnedPrepassPipeline->m_states.m_cullMode = renderCommand.m_cullMode;
					deferredSkinnedPrepassPipeline->m_states.m_lineWidth = renderCommand.m_lineWidth;
					deferredSkinnedPrepassPipeline->m_states.ApplyAllStates(commandBuffer);
					deferredSkinnedPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
					const auto skinnedMesh = renderCommand.m_skinnedMesh;
					graphics.m_drawCall[currentFrameIndex]++;
					graphics.m_triangles[currentFrameIndex] += renderCommand.m_skinnedMesh->m_skinnedTriangles.size();
					skinnedMesh->DrawIndexed(commandBuffer, deferredSkinnedPrepassPipeline->m_states, 1);
				}
			);

			vkCmdEndRendering(commandBuffer);
		}
		GeometryStorage::BindStrandPoints(commandBuffer);
		{
			const auto depthAttachment = camera->m_renderTexture->GetDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.pDepthAttachment = &depthAttachment;
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();

			const auto& deferredStrandsPrepassPipeline = Graphics::GetGraphicsPipeline("STANDARD_STRANDS_DEFERRED_PREPASS");
			deferredStrandsPrepassPipeline->m_states.ResetAllStates(commandBuffer, colorAttachmentInfos.size());
			deferredStrandsPrepassPipeline->m_states.m_viewPort = viewport;
			deferredStrandsPrepassPipeline->m_states.m_scissor = scissor;
			
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredStrandsPrepassPipeline->Bind(commandBuffer);
			deferredStrandsPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			m_deferredStrandsRenderInstances.Dispatch([&](const StrandsRenderInstance& renderCommand)
				{
					RenderInstancePushConstant pushConstant;
					pushConstant.m_cameraIndex = cameraIndex;
					pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
					deferredStrandsPrepassPipeline->m_states.m_polygonMode = renderCommand.m_polygonMode;
					deferredStrandsPrepassPipeline->m_states.m_cullMode = renderCommand.m_cullMode;
					deferredStrandsPrepassPipeline->m_states.m_lineWidth = renderCommand.m_lineWidth;
					deferredStrandsPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
					const auto strands = renderCommand.m_strands;
					graphics.m_drawCall[currentFrameIndex]++;
					graphics.m_strandsSegments[currentFrameIndex] += renderCommand.m_strands->m_segments.size();
					strands->DrawIndexed(commandBuffer, deferredStrandsPrepassPipeline->m_states, 1);
				}
			);

			vkCmdEndRendering(commandBuffer);
		}

#pragma endregion
#pragma region Lighting pass
		GeometryStorage::BindVertices(commandBuffer);
		{
			camera->TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			camera->m_renderTexture->GetDepthImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->GetRenderTexture()->AppendColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();
			renderInfo.pDepthAttachment = VK_NULL_HANDLE;
			m_lighting->m_directionalLightShadowMap->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			const auto& deferredLightingPipeline = isSceneCamera ? Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA") : Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_LIGHTING");
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredLightingPipeline->m_states.ResetAllStates(commandBuffer, colorAttachmentInfos.size());
			deferredLightingPipeline->m_states.m_depthTest = false;
			
			deferredLightingPipeline->Bind(commandBuffer);
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 1, camera->m_gBufferDescriptorSet->GetVkDescriptorSet());
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 2, m_lighting->m_lightingDescriptorSet->GetVkDescriptorSet());
			deferredLightingPipeline->m_states.m_viewPort = viewport;
			deferredLightingPipeline->m_states.m_scissor = scissor;
			RenderInstancePushConstant pushConstant;
			pushConstant.m_cameraIndex = cameraIndex;
			pushConstant.m_lightSplitIndex = needFade ? glm::max(128, 256 - editorLayer->m_selectionAlpha) : 256;
			pushConstant.m_instanceIndex = needFade ? 1 : 0;
			deferredLightingPipeline->PushConstant(commandBuffer, 0, pushConstant);
			const auto mesh = Resources::GetResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");

			mesh->DrawIndexed(commandBuffer, deferredLightingPipeline->m_states, 1);
			vkCmdEndRendering(commandBuffer);
		}
#pragma endregion
		}
	);
#pragma endregion

#pragma region ForwardRendering

#pragma endregion

	//Post processing
	if (const auto postProcessingStack = camera->m_postProcessingStack.Get<PostProcessingStack>())
	{
		postProcessingStack->Process(camera);
	}

	camera->m_rendered = true;
	camera->m_requireRendering = false;
}

void RenderLayer::DrawMesh(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Material>& material,
	glm::mat4 model, bool castShadow)
{
	auto scene = Application::GetActiveScene();
	MaterialInfoBlock materialInfoBlock;
	material->UpdateMaterialInfoBlock(materialInfoBlock);
	auto materialIndex = RegisterMaterialIndex(material->GetHandle(), materialInfoBlock);
	InstanceInfoBlock instanceInfoBlock;
	instanceInfoBlock.m_model.m_value = model;
	instanceInfoBlock.m_materialIndex = materialIndex;
	instanceInfoBlock.m_entitySelected = 0;
	instanceInfoBlock.m_meshletIndexOffset = mesh->m_meshletRange->m_offset;
	instanceInfoBlock.m_meshletSize = mesh->m_meshletRange->m_size;

	auto entityHandle = Handle();
	auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);
	RenderInstance renderInstance;
	renderInstance.m_commandType = RenderCommandType::FromAPI;
	renderInstance.m_owner = Entity();
	renderInstance.m_mesh = mesh;
	renderInstance.m_castShadow = castShadow;
	renderInstance.m_meshletSize = mesh->m_meshletRange->m_size;
	renderInstance.m_instanceIndex = instanceIndex;
	if (instanceInfoBlock.m_entitySelected == 1) m_needFade = true;
	if (material->m_drawSettings.m_blending)
	{
		m_transparentRenderInstances.m_renderCommands.push_back(renderInstance);
	}
	else
	{
		m_deferredRenderInstances.m_renderCommands.push_back(renderInstance);
	}

	auto& newMeshTask = m_meshDrawMeshTasksIndirectCommands.emplace_back();
	newMeshTask.groupCountX = 1;
	newMeshTask.groupCountY = 1;
	newMeshTask.groupCountZ = 1;

	auto& newDrawTask = m_meshDrawIndexedIndirectCommands.emplace_back();
	newDrawTask.instanceCount = 1;
	newDrawTask.firstIndex = mesh->m_triangleRange->m_offset * 3;
	newDrawTask.indexCount = static_cast<uint32_t>(mesh->m_triangles.size() * 3);
	newDrawTask.vertexOffset = 0;
	newDrawTask.firstInstance = 0;

	m_totalMeshTriangles += mesh->m_triangles.size();
}

const std::shared_ptr<DescriptorSet>& RenderLayer::GetPerFrameDescriptorSet() const
{
	return m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()];
}
