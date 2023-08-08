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
#include "Cubemap.hpp"
#include "Jobs.hpp"
#include "MeshStorage.hpp"
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

void RenderLayer::OnCreate()
{
	CreateStandardDescriptorBuffers();
	UpdateStandardBindings();

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
	const auto scene = GetScene();
	if (!scene) return;

	ApplyAnimator();

	m_deferredRenderInstances.m_renderCommands.clear();
	m_deferredSkinnedRenderInstances.m_renderCommands.clear();
	m_deferredInstancedRenderInstances.m_renderCommands.clear();
	m_transparentRenderInstances.m_renderCommands.clear();
	m_transparentSkinnedRenderInstances.m_renderCommands.clear();
	m_transparentInstancedRenderInstances.m_renderCommands.clear();

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

	Bound worldBound;
	std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>> cameras;
	CollectCameras(cameras);
	CollectRenderInstances(worldBound);
	scene->SetBound(worldBound);

	CollectDirectionalLights(cameras);
	CollectPointLights();
	CollectSpotLights();

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

	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	m_renderInfoDescriptorBuffers[currentFrameIndex]->Upload(m_renderInfoBlock);
	m_environmentInfoDescriptorBuffers[currentFrameIndex]->Upload(m_environmentInfoBlock);
	m_directionalLightInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_directionalLightInfoBlocks);
	m_pointLightInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_pointLightInfoBlocks);
	m_spotLightInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_spotLightInfoBlocks);

	m_cameraInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_cameraInfoBlocks);
	m_materialInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_materialInfoBlocks);
	m_instanceInfoDescriptorBuffers[currentFrameIndex]->UploadVector(m_instanceInfoBlocks);

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
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(6, bufferInfo);
	bufferInfo.buffer = m_directionalLightInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(7, bufferInfo);
	bufferInfo.buffer = m_pointLightInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(8, bufferInfo);
	bufferInfo.buffer = m_spotLightInfoDescriptorBuffers[currentFrameIndex]->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(9, bufferInfo);

	bufferInfo.buffer = MeshStorage::GetVertexBuffer()->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(10, bufferInfo);
	bufferInfo.buffer = MeshStorage::GetMeshletBuffer()->GetVkBuffer();
	m_perFrameDescriptorSets[currentFrameIndex]->UpdateBufferDescriptorBinding(11, bufferInfo);

	PreparePointAndSpotLightShadowMap();

	for (const auto& [cameraGlobalTransform, camera] : cameras)
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
		throw std::runtime_error("Unable to find instance!");
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
		m_directionalLightInfoBlocks.resize(Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE * cameras.size());
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
		Lighting::AllocateAtlas(m_renderInfoBlock.m_directionalLightSize, Graphics::Constants::DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION, viewPortResults);
		for (const auto& [cameraGlobalTransform, camera] : cameras) {
			auto cameraIndex = GetCameraIndex(camera->GetHandle());
			for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
			{
				const auto blockIndex = cameraIndex * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE + i;
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
				const auto blockIndex = cameraIndex * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE + directionalLightIndex;
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
					GLfloat storedW = shadowOrigin.w;
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
		Lighting::AllocateAtlas(m_renderInfoBlock.m_pointLightSize, Graphics::Constants::POINT_LIGHT_SHADOW_MAP_RESOLUTION, viewPortResults);
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
		Lighting::AllocateAtlas(m_renderInfoBlock.m_spotLightSize, Graphics::Constants::SPOT_LIGHT_SHADOW_MAP_RESOLUTION, viewPortResults);
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
		std::vector<std::shared_future<void>> results;

		Jobs::ParallelFor(owners->size(), [&](unsigned i)
			{
				const auto entity = owners->at(i);
				if (!scene->IsEntityEnabled(entity)) return;
				const auto animator = scene->GetOrSetPrivateComponent<Animator>(owners->at(i)).lock();
				if (!animator->IsEnabled()) return;
				animator->Apply();
			}, results);
		for (const auto& i : results)
			i.wait();
	}
	if (const auto* owners =
		scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>())
	{
		std::vector<std::shared_future<void>> results;
		Jobs::ParallelFor(owners->size(), [&](unsigned i)
			{
				const auto entity = owners->at(i);
				if (!scene->IsEntityEnabled(entity)) return;
				const auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(entity).lock();
				if (!skinnedMeshRenderer->IsEnabled()) return;
				skinnedMeshRenderer->UpdateBoneMatrices();
			}, results);
		for (const auto& i : results)
			i.wait();

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
	const auto& pointLightShadowPipeline = Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP");
	const auto& spotLightShadowPipeline = Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP");

	const auto& pointLightShadowSkinnedPipeline = Graphics::GetGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_SKINNED");
	const auto& spotLightShadowSkinnedPipeline = Graphics::GetGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_SKINNED");
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
				{
					VkRenderingInfo renderInfo{};
					auto depthAttachment = m_lighting->GetLayeredPointLightDepthAttachmentInfo(face, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
					renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
					renderInfo.renderArea = renderArea;
					renderInfo.layerCount = 1;
					renderInfo.colorAttachmentCount = 0;
					renderInfo.pColorAttachments = nullptr;
					renderInfo.pDepthAttachment = &depthAttachment;
					pointLightShadowPipeline->m_states.m_colorBlendAttachmentStates.clear();
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
						m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
							{
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = i;
								pushConstant.m_lightSplitIndex = face;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								pointLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
								const auto mesh = renderCommand.m_mesh;
								mesh->Bind(commandBuffer);
								mesh->DrawIndexed(commandBuffer, pointLightShadowPipeline->m_states);
							}
						);
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
					pointLightShadowSkinnedPipeline->m_states.m_colorBlendAttachmentStates.clear();
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
								pointLightShadowSkinnedPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->m_descriptorSet->GetVkDescriptorSet());
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = i;
								pushConstant.m_lightSplitIndex = face;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								pointLightShadowSkinnedPipeline->PushConstant(commandBuffer, 0, pushConstant);
								const auto skinnedMesh = renderCommand.m_skinnedMesh;
								skinnedMesh->Bind(commandBuffer);
								skinnedMesh->DrawIndexed(commandBuffer, pointLightShadowSkinnedPipeline->m_states);
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

			{
				VkRenderingInfo renderInfo{};
				auto depthAttachment = m_lighting->GetSpotLightDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.colorAttachmentCount = 0;
				renderInfo.pColorAttachments = nullptr;
				renderInfo.pDepthAttachment = &depthAttachment;
				spotLightShadowPipeline->m_states.m_colorBlendAttachmentStates.clear();
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
					m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
						{
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = i;
							pushConstant.m_lightSplitIndex = 0;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							spotLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto mesh = renderCommand.m_mesh;
							mesh->Bind(commandBuffer);
							mesh->DrawIndexed(commandBuffer, spotLightShadowPipeline->m_states);
						}
					);
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
				spotLightShadowSkinnedPipeline->m_states.m_colorBlendAttachmentStates.clear();
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
							spotLightShadowSkinnedPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->m_descriptorSet->GetVkDescriptorSet());
							RenderInstancePushConstant pushConstant;
							pushConstant.m_cameraIndex = i;
							pushConstant.m_lightSplitIndex = 0;
							pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
							spotLightShadowSkinnedPipeline->PushConstant(commandBuffer, 0, pushConstant);
							const auto skinnedMesh = renderCommand.m_skinnedMesh;
							skinnedMesh->Bind(commandBuffer);
							skinnedMesh->DrawIndexed(commandBuffer, spotLightShadowSkinnedPipeline->m_states);
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
			instanceInfoBlock.m_materialIndex = materialIndex;
			instanceInfoBlock.m_infoIndex1 = scene->IsEntityAncestorSelected(owner) ? 1 : 0;
			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);
			RenderInstance renderInstance;
			renderInstance.m_owner = owner;
			renderInstance.m_mesh = mesh;
			renderInstance.m_castShadow = mmc->m_castShadow;

			renderInstance.m_instanceIndex = instanceIndex;
			if (instanceInfoBlock.m_infoIndex1 == 1) m_needFade = true;
			if (material->m_drawSettings.m_blending)
			{
				m_transparentRenderInstances.m_renderCommands.push_back(renderInstance);
			}
			else
			{
				m_deferredRenderInstances.m_renderCommands.push_back(renderInstance);
			}
		}
	}
	if (const auto* owners = scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>())
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

			MaterialInfoBlock materialInfoBlock;
			material->UpdateMaterialInfoBlock(materialInfoBlock);
			auto materialIndex = RegisterMaterialIndex(material->GetHandle(), materialInfoBlock);
			InstanceInfoBlock instanceInfoBlock;
			instanceInfoBlock.m_model = gt;
			instanceInfoBlock.m_materialIndex = materialIndex;
			instanceInfoBlock.m_infoIndex1 = scene->IsEntityAncestorSelected(owner) ? 1 : 0;

			auto entityHandle = scene->GetEntityHandle(owner);
			auto instanceIndex = RegisterInstanceIndex(entityHandle, instanceInfoBlock);

			SkinnedRenderInstance renderInstance;
			renderInstance.m_owner = owner;
			renderInstance.m_skinnedMesh = skinnedMesh;
			renderInstance.m_castShadow = smmc->m_castShadow;
			renderInstance.m_boneMatrices = smmc->m_boneMatrices;

			renderInstance.m_instanceIndex = instanceIndex;
			if (instanceInfoBlock.m_infoIndex1 == 1) m_needFade = true;

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
				renderInstance.m_mesh = mesh;
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
				renderInstance.m_mesh = strands;
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
	m_instanceInfoDescriptorBuffers.clear();
	m_kernelDescriptorBuffers.clear();
	m_directionalLightInfoDescriptorBuffers.clear();
	m_pointLightInfoDescriptorBuffers.clear();
	m_spotLightInfoDescriptorBuffers.clear();

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
		bufferCreateInfo.size = sizeof(DirectionalLightInfo) * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE * Graphics::Constants::INITIAL_CAMERA_SIZE;
		m_directionalLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(PointLightInfo) * Graphics::Constants::MAX_POINT_LIGHT_SIZE;
		m_pointLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(SpotLightInfo) * Graphics::Constants::MAX_SPOT_LIGHT_SIZE;
		m_spotLightInfoDescriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
	}
#pragma endregion
}



void RenderLayer::UpdateStandardBindings()
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();

	m_perFrameDescriptorSets.clear();
	for (size_t i = 0; i < maxFramesInFlight; i++)
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
		bufferInfo.buffer = m_instanceInfoDescriptorBuffers[i]->GetVkBuffer();
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
			mesh->Bind(commandBuffer);
			mesh->DrawIndexed(commandBuffer, environmentalBrdfPipeline->m_states);
			vkCmdEndRendering(commandBuffer);
#pragma endregion
		}
		m_environmentalBRDFLut->m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	);
}




void RenderLayer::RenderToCamera(const GlobalTransform& cameraGlobalTransform, const std::shared_ptr<Camera>& camera)
{
	const int cameraIndex = GetCameraIndex(camera->GetHandle());
	const auto scene = Application::GetActiveScene();
	const auto& directionalLightShadowPipeline = Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP");
	const auto& directionalLightShadowPipelineSkinned = Graphics::GetGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED");

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

					for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
					{
						const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE + i];
						viewport.x = directionalLightInfoBlock.m_viewPort.x;
						viewport.y = directionalLightInfoBlock.m_viewPort.y;
						viewport.width = directionalLightInfoBlock.m_viewPort.z;
						viewport.height = directionalLightInfoBlock.m_viewPort.w;
						scissor.extent.width = directionalLightInfoBlock.m_viewPort.z;
						scissor.extent.height = directionalLightInfoBlock.m_viewPort.w;
						directionalLightShadowPipeline->m_states.m_scissor = scissor;
						directionalLightShadowPipeline->m_states.m_viewPort = viewport;
						directionalLightShadowPipeline->m_states.ApplyAllStates(commandBuffer);

						m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
							{
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = cameraIndex * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE + i;
								pushConstant.m_lightSplitIndex = split;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								directionalLightShadowPipeline->PushConstant(commandBuffer, 0, pushConstant);
								const auto mesh = renderCommand.m_mesh;
								mesh->Bind(commandBuffer);
								mesh->DrawIndexed(commandBuffer, directionalLightShadowPipeline->m_states);

							}
						);

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
					directionalLightShadowPipelineSkinned->m_states.m_scissor = scissor;
					directionalLightShadowPipelineSkinned->m_states.m_viewPort = viewport;
					directionalLightShadowPipelineSkinned->m_states.m_colorBlendAttachmentStates.clear();

					vkCmdBeginRendering(commandBuffer, &renderInfo);
					directionalLightShadowPipelineSkinned->Bind(commandBuffer);
					directionalLightShadowPipelineSkinned->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());

					for (int i = 0; i < m_renderInfoBlock.m_directionalLightSize; i++)
					{
						const auto& directionalLightInfoBlock = m_directionalLightInfoBlocks[cameraIndex * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE + i];
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
								directionalLightShadowPipelineSkinned->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->m_descriptorSet->GetVkDescriptorSet());
								RenderInstancePushConstant pushConstant;
								pushConstant.m_cameraIndex = cameraIndex * Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE + i;
								pushConstant.m_lightSplitIndex = split;
								pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
								directionalLightShadowPipelineSkinned->PushConstant(commandBuffer, 0, pushConstant);
								const auto skinnedMesh = renderCommand.m_skinnedMesh;
								skinnedMesh->Bind(commandBuffer);
								skinnedMesh->DrawIndexed(commandBuffer, directionalLightShadowPipelineSkinned->m_states);
							}
						);

					}
					vkCmdEndRendering(commandBuffer);
				}
			}
		}
	);

	const auto editorLayer = Application::GetLayer<EditorLayer>();
	bool isSceneCamera = false;
	bool needFade = false;
	if (editorLayer)
	{
		if (camera.get() == editorLayer->m_sceneCamera.get()) isSceneCamera = true;
		if (m_needFade) needFade = true;
	}

	const auto& deferredLightingPipeline = isSceneCamera ? Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA") : Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_LIGHTING");
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
		camera->m_renderTexture->m_depthImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);

		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = renderArea;
		renderInfo.layerCount = 1;
#pragma endregion
#pragma region Geometry pass
		{
			const auto depthAttachment = camera->m_renderTexture->GetDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.pDepthAttachment = &depthAttachment;
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();

			const auto& deferredPrepassPipeline = Graphics::GetGraphicsPipeline("STANDARD_DEFERRED_PREPASS");
			deferredPrepassPipeline->m_states.m_viewPort = viewport;
			deferredPrepassPipeline->m_states.m_scissor = scissor;
			deferredPrepassPipeline->m_states.m_colorBlendAttachmentStates.clear();
			deferredPrepassPipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
			for (auto& i : deferredPrepassPipeline->m_states.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				i.blendEnable = VK_FALSE;
			}
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredPrepassPipeline->Bind(commandBuffer);
			deferredPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			m_deferredRenderInstances.Dispatch([&](const RenderInstance& renderCommand)
				{
					RenderInstancePushConstant pushConstant;
					pushConstant.m_cameraIndex = cameraIndex;
					pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
					deferredPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
					const auto mesh = renderCommand.m_mesh;
					mesh->Bind(commandBuffer);
					mesh->DrawIndexed(commandBuffer, deferredPrepassPipeline->m_states);
				}
				);

			vkCmdEndRendering(commandBuffer);
		}
		{
			const auto depthAttachment = camera->m_renderTexture->GetDepthAttachmentInfo(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.pDepthAttachment = &depthAttachment;
			std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
			camera->AppendGBufferColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
			renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
			renderInfo.pColorAttachments = colorAttachmentInfos.data();

			const auto& deferredSkinnedPrepassPipeline = Graphics::GetGraphicsPipeline("STANDARD_SKINNED_DEFERRED_PREPASS");
			deferredSkinnedPrepassPipeline->m_states.m_viewPort = viewport;
			deferredSkinnedPrepassPipeline->m_states.m_scissor = scissor;
			deferredSkinnedPrepassPipeline->m_states.m_colorBlendAttachmentStates.clear();
			deferredSkinnedPrepassPipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
			for (auto& i : deferredSkinnedPrepassPipeline->m_states.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				i.blendEnable = VK_FALSE;
			}
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredSkinnedPrepassPipeline->Bind(commandBuffer);
			deferredSkinnedPrepassPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			m_deferredSkinnedRenderInstances.Dispatch([&](const SkinnedRenderInstance& renderCommand)
				{
					deferredSkinnedPrepassPipeline->BindDescriptorSet(commandBuffer, 1, renderCommand.m_boneMatrices->m_descriptorSet->GetVkDescriptorSet());
					RenderInstancePushConstant pushConstant;
					pushConstant.m_cameraIndex = cameraIndex;
					pushConstant.m_instanceIndex = renderCommand.m_instanceIndex;
					deferredSkinnedPrepassPipeline->PushConstant(commandBuffer, 0, pushConstant);
					const auto skinnedMesh = renderCommand.m_skinnedMesh;
					skinnedMesh->Bind(commandBuffer);
					skinnedMesh->DrawIndexed(commandBuffer, deferredSkinnedPrepassPipeline->m_states);
				}
				);

			vkCmdEndRendering(commandBuffer);
		}
#pragma endregion
#pragma region Lighting pass
		{
			camera->TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			camera->m_renderTexture->m_depthImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			deferredLightingPipeline->m_states.m_depthTest = false;
			deferredLightingPipeline->m_states.m_colorBlendAttachmentStates.clear();
			deferredLightingPipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
			for (auto& i : deferredLightingPipeline->m_states.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				i.blendEnable = VK_FALSE;
			}
			deferredLightingPipeline->Bind(commandBuffer);
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]->GetVkDescriptorSet());
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 1, camera->m_gBufferDescriptorSet->GetVkDescriptorSet());
			deferredLightingPipeline->BindDescriptorSet(commandBuffer, 2, m_lighting->m_lightingDescriptorSet->GetVkDescriptorSet());
			deferredLightingPipeline->m_states.m_viewPort = viewport;
			deferredLightingPipeline->m_states.m_scissor = scissor;
			RenderInstancePushConstant pushConstant;
			pushConstant.m_cameraIndex = cameraIndex;
			pushConstant.m_lightSplitIndex = needFade ? glm::max(32, 256 - editorLayer->m_selectionAlpha) : 256;
			pushConstant.m_instanceIndex = needFade ? 1 : 0;
			deferredLightingPipeline->PushConstant(commandBuffer, 0, pushConstant);
			const auto mesh = Resources::GetResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");
			mesh->Bind(commandBuffer);
			mesh->DrawIndexed(commandBuffer, deferredLightingPipeline->m_states, false);
			vkCmdEndRendering(commandBuffer);
		}
#pragma endregion
		}
	);


	camera->m_rendered = true;
	camera->m_requireRendering = false;
}
