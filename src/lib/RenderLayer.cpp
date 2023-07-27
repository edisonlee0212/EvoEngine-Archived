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

glm::vec3 CameraInfoBlock::Project(const glm::vec3& position) const
{
	return m_projection * m_view * glm::vec4(position, 1.0f);
}

glm::vec3 CameraInfoBlock::UnProject(const glm::vec3& position) const
{
	const glm::mat4 inverse = glm::inverse(m_projection * m_view);
	auto start = glm::vec4(position, 1.0f);
	start = inverse * start;
	return start / start.w;
}



void RenderTask::Dispatch(
	const std::function<void(const std::shared_ptr<Material>&)>&
	beginCommandGroupAction,
	const std::function<void(const RenderCommand&)>& commandAction) const
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
	CreateGraphicsPipelines();

	if(const auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		editorLayer->m_sceneCamera = Serialization::ProduceSerializable<Camera>();
		editorLayer->m_sceneCamera->m_clearColor = glm::vec3(59.0f / 255.0f, 85 / 255.0f, 143 / 255.f);
		editorLayer->m_sceneCamera->m_useClearColor = false;
		editorLayer->m_sceneCamera->OnCreate();
	}
}

void RenderLayer::OnDestroy()
{
	
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

	Graphics::UploadRenderInfo(m_renderInfoBlock);
	Graphics::UploadEnvironmentInfo(m_environmentInfoBlock);

	Bound worldBound;
	std::vector<std::shared_ptr<Camera>> cameras;
	CollectRenderTasks(worldBound, cameras);
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
			RenderToCamera(camera, scene->GetDataComponent<GlobalTransform>(camera->GetOwner()));
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
	standardDeferredPrepass->InitializeStandardBindings();
	standardDeferredPrepass->PrepareLayouts();
	standardDeferredPrepass->UpdateStandardBindings();

	standardDeferredPrepass->m_depthAttachmentFormat = Graphics::ImageFormats::m_gBufferDepth;
	standardDeferredPrepass->m_stencilAttachmentFormat = Graphics::ImageFormats::m_gBufferDepth;
	standardDeferredPrepass->m_colorAttachmentFormats = { 3, Graphics::ImageFormats::m_gBufferColor };

	standardDeferredPrepass->PreparePipeline();
	m_graphicsPipelines["STANDARD_DEFERRED_PREPASS"] = standardDeferredPrepass;

	/*
	auto standardDeferredLighting = std::make_shared<GraphicsPipeline>();
	standardDeferredLighting->m_vertexShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("TEXTURE_PASS_THROUGH_VERT"));
	standardDeferredLighting->m_fragmentShader = std::dynamic_pointer_cast<Shader>(Resources::GetResource("STANDARD_DEFERRED_LIGHTING_FRAG"));
	standardDeferredLighting->InitializeStandardBindings();
	standardDeferredLighting->PrepareLayouts();
	standardDeferredLighting->UpdateStandardBindings();
	standardDeferredLighting->PreparePipeline();
	m_graphicsPipelines["STANDARD_DEFERRED_LIGHTING"] = standardDeferredLighting;*/
}

void RenderLayer::CollectRenderTasks(Bound& worldBound, std::vector<std::shared_ptr<Camera>>& cameras)
{
	auto scene = GetScene();
	std::vector<std::pair<std::shared_ptr<Camera>, glm::vec3>> cameraPairs;
	if (auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		auto& sceneCamera = editorLayer->GetSceneCamera();
		if (sceneCamera && sceneCamera->IsEnabled())
		{
			cameraPairs.emplace_back(sceneCamera, editorLayer->m_sceneCameraPosition);
		}
	}
	if (const std::vector<Entity>* cameraEntities =
		scene->UnsafeGetPrivateComponentOwnersList<Camera>())
	{
		for (const auto& i : *cameraEntities)
		{
			if (!scene->IsEntityEnabled(i))
				continue;
			assert(scene->HasPrivateComponent<Camera>(i));
			auto camera = scene->GetOrSetPrivateComponent<Camera>(i).lock();
			if (!camera || !camera->IsEnabled())
				continue;
			cameraPairs.emplace_back(camera, scene->GetDataComponent<GlobalTransform>(i).GetPosition());
			cameras.emplace_back(camera);
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

				RenderCommand renderInstance;
				renderInstance.m_owner = owner;
				renderInstance.m_globalTransform = gt;
				renderInstance.m_renderGeometry = mesh;
				renderInstance.m_castShadow = mmc->m_castShadow;
				renderInstance.m_receiveShadow = mmc->m_receiveShadow;
				renderInstance.m_geometryType = RenderGeometryType::Mesh;
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

				RenderCommand renderInstance;
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

				RenderCommand renderInstance;
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

				RenderCommand renderInstance;
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

void RenderLayer::RenderToCamera(const std::shared_ptr<Camera>& camera, const GlobalTransform& cameraModel)
{
	
	CameraInfoBlock cameraInfoBlock = {};
	camera->UpdateCameraInfoBlock(cameraInfoBlock, cameraModel);
	Graphics::UploadCameraInfo(cameraInfoBlock);
	auto deferredPrepassProgram = m_graphicsPipelines["STANDARD_DEFERRED_PREPASS"];
	auto deferredLightingProgram = m_graphicsPipelines["STANDARD_DEFERRED_LIGHTING"];
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
				MaterialInfoBlock materialInfoBlock = {};
				material->m_drawSettings.ApplySettings(globalPipelineState);
				material->UpdateMaterialInfoBlock(materialInfoBlock);
				//We should also bind textures here.
			}, [&](const RenderCommand& renderCommand)
			{
				switch (renderCommand.m_geometryType)
				{
				case RenderGeometryType::Mesh: {
					ObjectInfoBlock objectInfoBlock{};
					objectInfoBlock.m_model = renderCommand.m_globalTransform.m_value;
					Graphics::UploadObjectInfo(objectInfoBlock);
					globalPipelineState.ApplyAllStates(commandBuffer);
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
