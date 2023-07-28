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

void RenderTask::Dispatch(
	const std::function<void(const std::shared_ptr<Material>&)>&
	beginCommandGroupAction,
	const std::function<void(const RenderInstance&)>& commandAction) const
{
	for (const auto& renderCollection : m_renderCommandsGroups)
	{
		const auto& material = renderCollection.second.m_material;
		beginCommandGroupAction(material);
		for (const auto& renderCommands : renderCollection.second.m_renderCommands)
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
	CreateStandardDescriptorBuffers();
	PrepareMaterialLayout();
	PreparePerFrameLayout();

	CreateGraphicsPipelines();

	if (const auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		editorLayer->m_sceneCamera = Serialization::ProduceSerializable<Camera>();
		editorLayer->m_sceneCamera->m_clearColor = glm::vec3(59.0f / 255.0f, 85 / 255.0f, 143 / 255.f);
		editorLayer->m_sceneCamera->m_useClearColor = false;
		editorLayer->m_sceneCamera->OnCreate();
	}
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
	m_deferredRenderInstances.clear();
	m_deferredInstancedRenderInstances.clear();
	m_forwardRenderInstances.clear();
	m_forwardInstancedRenderInstances.clear();
	m_transparentRenderInstances.clear();
	m_instancedTransparentRenderInstances.clear();

	m_cameraIndices.clear();
	m_materialIndices.clear();
	m_instanceIndices.clear();
	m_cameraInfoBlocks.clear();
	m_materialInfoBlocks.clear();
	m_instanceInfoBlocks.clear();

	UploadRenderInfoBlock(m_renderInfoBlock);
	UploadEnvironmentInfoBlock(m_environmentInfoBlock);

	Bound worldBound;
	std::vector<std::shared_ptr<Camera>> cameras;
	CollectRenderTasks(worldBound, cameras);

	UploadCameraInfoBlocks(m_cameraInfoBlocks);
	UploadMaterialInfoBlocks(m_materialInfoBlocks);
	UploadInstanceInfoBlocks(m_instanceInfoBlocks);

	scene->SetBound(worldBound);

	if (const std::shared_ptr<Camera> mainCamera = scene->m_mainCamera.Get<Camera>())
	{
		if (m_allowAutoResize) mainCamera->Resize({ m_mainCameraResolutionX, m_mainCameraResolutionY });
	}
	for (const auto& camera : cameras)
	{
		camera->m_rendered = false;
		if (camera->m_requireRendering)
		{
			RenderToCamera(camera);
		}
	}
}

void RenderLayer::LateUpdate()
{
}

void RenderLayer::CreateGraphicsPipelines()
{
	auto standardDeferredPrepass = std::make_shared<GraphicsPipeline>();
	standardDeferredPrepass->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_VERT"));
	standardDeferredPrepass->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_DEFERRED_FRAG"));

	standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(m_perFrameLayout);
	standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(m_materialLayout);

	standardDeferredPrepass->m_depthAttachmentFormat = Graphics::ImageFormats::m_gBufferDepth;
	standardDeferredPrepass->m_stencilAttachmentFormat = Graphics::ImageFormats::m_gBufferDepth;
	standardDeferredPrepass->m_colorAttachmentFormats = { 3, Graphics::ImageFormats::m_gBufferColor };

	auto& pushConstantRange = standardDeferredPrepass->m_pushConstantRanges.emplace_back();
	pushConstantRange.size = sizeof(RenderInstancePushConstant);
	pushConstantRange.offset = 0;
	pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

	standardDeferredPrepass->PreparePipeline();
	m_graphicsPipelines["STANDARD_DEFERRED_PREPASS"] = standardDeferredPrepass;

	/*
	auto standardDeferredLighting = std::make_shared<GraphicsPipeline>();
	standardDeferredLighting->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("TEXTURE_PASS_THROUGH_VERT"));
	standardDeferredLighting->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_DEFERRED_LIGHTING_FRAG"));
	standardDeferredLighting->InitializeStandardBindings();
	standardDeferredLighting->PreparePerPassLayouts();
	standardDeferredLighting->UpdateStandardBindings();
	standardDeferredLighting->PreparePipeline();
	m_graphicsPipelines["STANDARD_DEFERRED_LIGHTING"] = standardDeferredLighting;*/
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
	if(upload)
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

void RenderLayer::CollectRenderTasks(Bound& worldBound, std::vector<std::shared_ptr<Camera>>& cameras)
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

			cameras.push_back(sceneCamera);
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

			cameras.push_back(camera);
		}
	}
	auto& minBound = worldBound.m_min;
	auto& maxBound = worldBound.m_max;
	minBound = glm::vec3(INT_MAX);
	maxBound = glm::vec3(INT_MIN);

	const std::vector<Entity>* owners =
		scene->UnsafeGetPrivateComponentOwnersList<MeshRenderer>();
	if (owners)
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
				renderInstance.m_renderGeometry = mesh;
				renderInstance.m_castShadow = mmc->m_castShadow;
				renderInstance.m_receiveShadow = mmc->m_receiveShadow;
				renderInstance.m_geometryType = RenderGeometryType::Mesh;

				renderInstance.m_pushConstant.m_cameraIndex = GetCameraIndex(pair.first->GetHandle());
				renderInstance.m_pushConstant.m_materialIndex = materialIndex;
				renderInstance.m_pushConstant.m_instanceIndex = instanceIndex;

				if (material->m_drawSettings.m_blending)
				{
					auto& group = transparentRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);

				}
				else if (mmc->m_forwardRendering)
				{
					auto& group = forwardRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
				}
				else
				{
					auto& group = deferredRenderInstances.m_renderCommandsGroups[material->GetHandle()];
					group.m_material = material;
					group.m_renderCommands[mesh->GetHandle()].push_back(renderInstance);
				}
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

		vmaMapMemory(Graphics::GetVmaAllocator(), m_renderInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_renderInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_environmentInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_environmentalInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_cameraInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_cameraInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_materialInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_materialInfoBlockMemory[i])));
		vmaMapMemory(Graphics::GetVmaAllocator(), m_objectInfoDescriptorBuffers[i]->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_instanceInfoBlockMemory[i])));
	}
#pragma endregion
}

void RenderLayer::UpdateStandardBindings()
{
	std::vector<VkDescriptorBufferInfo> tempBufferInfos;
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	tempBufferInfos.resize(maxFramesInFlight);
	for (auto& i : tempBufferInfos)
	{
		i.offset = 0;
		i.range = VK_WHOLE_SIZE;
	}
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		tempBufferInfos[i].buffer = m_renderInfoDescriptorBuffers[i]->GetVkBuffer();
	}
	UpdateBufferDescriptorBinding(0, tempBufferInfos);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		tempBufferInfos[i].buffer = m_environmentInfoDescriptorBuffers[i]->GetVkBuffer();
	}
	UpdateBufferDescriptorBinding(1, tempBufferInfos);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		tempBufferInfos[i].buffer = m_cameraInfoDescriptorBuffers[i]->GetVkBuffer();
	}
	UpdateBufferDescriptorBinding(2, tempBufferInfos);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		tempBufferInfos[i].buffer = m_materialInfoDescriptorBuffers[i]->GetVkBuffer();
	}
	UpdateBufferDescriptorBinding(3, tempBufferInfos);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		tempBufferInfos[i].buffer = m_objectInfoDescriptorBuffers[i]->GetVkBuffer();
	}
	UpdateBufferDescriptorBinding(4, tempBufferInfos);
}



void RenderLayer::ClearDescriptorSets()
{
	m_descriptorSetLayoutBindings.clear();

	m_perFrameLayout.reset();

	if (!m_perFrameDescriptorSets.empty()) vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perFrameDescriptorSets.size(), m_perFrameDescriptorSets.data());

	m_perFrameDescriptorSets.clear();

	m_layoutReady = false;
	m_descriptorSetsReady = false;
}

void RenderLayer::CheckDescriptorSetsReady()
{
	for (const auto& binding : m_descriptorSetLayoutBindings)
	{
		if (!binding.second.m_ready)
		{
			m_descriptorSetsReady = false;
		}
	}
	m_descriptorSetsReady = true;
}



void RenderLayer::PushDescriptorBinding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stageFlags)
{
	VkDescriptorSetLayoutBinding bindingInfo{};
	bindingInfo.binding = binding;
	bindingInfo.descriptorCount = 1;
	bindingInfo.descriptorType = type;
	bindingInfo.pImmutableSamplers = nullptr;
	bindingInfo.stageFlags = stageFlags;
	m_descriptorSetLayoutBindings[binding] = { bindingInfo, false };

	m_descriptorSetsReady = false;
	m_layoutReady = false;
}

void RenderLayer::UpdateImageDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorImageInfo>& imageInfos)
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	assert(maxFramesInFlight == imageInfos.size());
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		const auto& descriptorBinding = m_descriptorSetLayoutBindings[binding];
		VkWriteDescriptorSet writeInfo{};
		writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeInfo.dstSet = m_perFrameDescriptorSets[i];
		writeInfo.dstBinding = binding;
		writeInfo.dstArrayElement = 0;
		writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
		writeInfo.descriptorCount = descriptorBinding.m_binding.descriptorCount;
		writeInfo.pImageInfo = &imageInfos[i];
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
	}
	m_descriptorSetLayoutBindings[binding].m_ready = true;
	CheckDescriptorSetsReady();
}

void RenderLayer::UpdateBufferDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorBufferInfo>& bufferInfos)
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	assert(maxFramesInFlight == bufferInfos.size());
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		const auto& descriptorBinding = m_descriptorSetLayoutBindings[binding];
		VkWriteDescriptorSet writeInfo{};
		writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeInfo.dstSet = m_perFrameDescriptorSets[i];
		writeInfo.dstBinding = binding;
		writeInfo.dstArrayElement = 0;
		writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
		writeInfo.descriptorCount = descriptorBinding.m_binding.descriptorCount;
		writeInfo.pBufferInfo = &bufferInfos[i];
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
	}
	m_descriptorSetLayoutBindings[binding].m_ready = true;
	CheckDescriptorSetsReady();
}

void RenderLayer::PrepareMaterialLayout()
{
	std::vector<VkDescriptorSetLayoutBinding> materialBindings;

	VkDescriptorSetLayoutBinding bindingInfo{};
	bindingInfo.binding = 10;
	bindingInfo.descriptorCount = 1;
	bindingInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindingInfo.pImmutableSamplers = nullptr;
	bindingInfo.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	materialBindings.emplace_back(bindingInfo);

	bindingInfo.binding = 11;
	materialBindings.emplace_back(bindingInfo);
	bindingInfo.binding = 12;
	materialBindings.emplace_back(bindingInfo);
	bindingInfo.binding = 13;
	materialBindings.emplace_back(bindingInfo);
	bindingInfo.binding = 14;
	materialBindings.emplace_back(bindingInfo);

	VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
	materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialBindings.size());
	materialLayoutInfo.pBindings = materialBindings.data();
	m_materialLayout = std::make_shared<DescriptorSetLayout>(materialLayoutInfo);
}

void RenderLayer::PreparePerFrameLayout()
{
	if (!m_descriptorSetLayoutBindings.empty())
	{
		EVOENGINE_ERROR("Already contain bindings!");
		return;
	}
	ClearDescriptorSets();

	PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL);

	std::vector<VkDescriptorSetLayoutBinding> perFrameBindings;

	for (const auto& bindingPair : m_descriptorSetLayoutBindings)
	{
		perFrameBindings.push_back(bindingPair.second.m_binding);
	}

	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	VkDescriptorSetLayoutCreateInfo perFrameLayoutInfo{};
	perFrameLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perFrameLayoutInfo.bindingCount = static_cast<uint32_t>(perFrameBindings.size());
	perFrameLayoutInfo.pBindings = perFrameBindings.data();
	m_perFrameLayout = std::make_shared<DescriptorSetLayout>(perFrameLayoutInfo);


	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();

	const std::vector perFrameLayouts(maxFramesInFlight, m_perFrameLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perFrameLayouts.data();
	m_perFrameDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perFrameDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}
	m_layoutReady = true;

	UpdateStandardBindings();
}



void RenderLayer::RenderToCamera(const std::shared_ptr<Camera>& camera)
{
	const auto& deferredPrepassProgram = m_graphicsPipelines["STANDARD_DEFERRED_PREPASS"];
	//const auto& deferredLightingProgram = m_graphicsPipelines["STANDARD_DEFERRED_LIGHTING"];
	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GraphicsGlobalStates& globalPipelineState) {
		VkRect2D renderArea;
		renderArea.offset = { 0, 0 };
		renderArea.extent.width = camera->GetSize().x;
		renderArea.extent.height = camera->GetSize().y;
		std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
		camera->AppendColorAttachmentInfos(colorAttachmentInfos);
		const auto depthAttachment = camera->GetDepthAttachmentInfo();
		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = renderArea;
		renderInfo.layerCount = 1;
		renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
		renderInfo.pColorAttachments = colorAttachmentInfos.data();
		renderInfo.pDepthAttachment = &depthAttachment;
		vkCmdBeginRendering(commandBuffer, &renderInfo);

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
		deferredPrepassProgram->Bind(commandBuffer);
		deferredPrepassProgram->BindDescriptorSet(commandBuffer, 0, m_perFrameDescriptorSets[Graphics::GetCurrentFrameIndex()]);

		globalPipelineState.m_colorBlendAttachmentStates.clear();
		globalPipelineState.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
		for (auto& i : globalPipelineState.m_colorBlendAttachmentStates)
		{
			i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			i.blendEnable = VK_FALSE;
		}
		globalPipelineState.ApplyAllStates(commandBuffer, true);

		m_deferredRenderInstances[camera->GetHandle()].Dispatch([&](const std::shared_ptr<Material>& material)
			{
				deferredPrepassProgram->BindDescriptorSet(commandBuffer, 1, material->m_descriptorSet);
				//We should also bind textures here.
			}, [&](const RenderInstance& renderCommand)
			{
				switch (renderCommand.m_geometryType)
				{
				case RenderGeometryType::Mesh: {
					globalPipelineState.ApplyAllStates(commandBuffer);
					deferredPrepassProgram->PushConstant(commandBuffer, 0, renderCommand.m_pushConstant);
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
	);


	camera->m_rendered = true;
	camera->m_requireRendering = false;
}


void RenderLayer::UploadEnvironmentInfoBlock(const EnvironmentInfoBlock& environmentInfoBlock)
{
	memcpy(m_environmentalInfoBlockMemory[Graphics::GetCurrentFrameIndex()], &environmentInfoBlock, sizeof(EnvironmentInfoBlock));
}

void RenderLayer::UploadRenderInfoBlock(const RenderInfoBlock& renderInfoBlock)
{
	memcpy(m_renderInfoBlockMemory[Graphics::GetCurrentFrameIndex()], &renderInfoBlock, sizeof(RenderInfoBlock));
}

void RenderLayer::UploadCameraInfoBlocks(const std::vector<CameraInfoBlock>& cameraInfoBlocks)
{
	memcpy(m_cameraInfoBlockMemory[Graphics::GetCurrentFrameIndex()], cameraInfoBlocks.data(), sizeof(CameraInfoBlock) * cameraInfoBlocks.size());
}

void RenderLayer::UploadMaterialInfoBlocks(const std::vector<MaterialInfoBlock>& materialInfoBlocks)
{
	memcpy(m_materialInfoBlockMemory[Graphics::GetCurrentFrameIndex()], materialInfoBlocks.data(), sizeof(MaterialInfoBlock) * materialInfoBlocks.size());
}

void RenderLayer::UploadInstanceInfoBlocks(const std::vector<InstanceInfoBlock>& objectInfoBlocks)
{
	memcpy(m_instanceInfoBlockMemory[Graphics::GetCurrentFrameIndex()], objectInfoBlocks.data(), sizeof(InstanceInfoBlock) * objectInfoBlocks.size());
}


