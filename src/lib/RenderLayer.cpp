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
	CreateRenderPasses();
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
	m_renderPasses.clear();
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

void RenderLayer::CreateRenderPasses()
{
	if (auto editorLayer = Application::GetLayer<EditorLayer>())
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = Graphics::GetSwapchain()->GetImageFormat();
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = VK_FORMAT_D24_UNORM_S8_UINT;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		const std::vector<VkSubpassDescription> subpasses = { subpass };
		const std::vector<VkSubpassDependency> dependencies = { dependency };
		renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		const std::vector attachments = { colorAttachment, depthAttachment };
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		m_renderPasses.insert({ "SCREEN_PRESENT", std::make_unique<RenderPass>(renderPassInfo) });
	}

	{
		VkSubpassDescription geometricSubpass{};
		VkSubpassDescription shadingSubpass{};

		//Subpass 1: To gBuffer.
		VkAttachmentReference gBufferDepthAttachmentRef{};
		gBufferDepthAttachmentRef.attachment = 0;
		gBufferDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		VkAttachmentReference gBufferNormalAttachmentRef{};
		gBufferNormalAttachmentRef.attachment = 1;
		gBufferNormalAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		VkAttachmentReference gBufferAlbedoAttachmentRef{};
		gBufferAlbedoAttachmentRef.attachment = 2;
		gBufferAlbedoAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		VkAttachmentReference gBufferMaterialAttachmentRef{};
		gBufferMaterialAttachmentRef.attachment = 3;
		gBufferMaterialAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		std::vector colorReferences {gBufferNormalAttachmentRef, gBufferAlbedoAttachmentRef, gBufferMaterialAttachmentRef};

		geometricSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		geometricSubpass.colorAttachmentCount = colorReferences.size();
		geometricSubpass.pColorAttachments = colorReferences.data();
		geometricSubpass.pDepthStencilAttachment = &gBufferDepthAttachmentRef;


		//Subpass 2: To RenderTexture
		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 0;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 1;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		shadingSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		shadingSubpass.colorAttachmentCount = 1;
		shadingSubpass.pColorAttachments = &colorAttachmentRef;
		shadingSubpass.pDepthStencilAttachment = &depthAttachmentRef;


		VkSubpassDependency geometryPassDependency{};
		geometryPassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		geometryPassDependency.dstSubpass = 1;

		geometryPassDependency.srcStageMask = 0;
		geometryPassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

		geometryPassDependency.srcAccessMask = 0;
		geometryPassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkSubpassDependency shadingPassDependency{};
		shadingPassDependency.srcSubpass = 0;
		shadingPassDependency.dstSubpass = 1;
		shadingPassDependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		shadingPassDependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		shadingPassDependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		shadingPassDependency.dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;

		const std::vector subpasses = { geometricSubpass, shadingSubpass };
		const std::vector dependencies = { geometryPassDependency,  shadingPassDependency };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();
		auto attachmentDescriptions = Camera::GetAttachmentDescriptions();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescriptions.size());
		renderPassInfo.pAttachments = attachmentDescriptions.data();
		m_renderPasses.insert({ "DEFERRED_RENDERING", std::make_unique<RenderPass>(renderPassInfo) });
	}
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

const std::unique_ptr<RenderPass>& RenderLayer::GetRenderPass(const std::string& name)
{
	return m_renderPasses.at(name);
}

void RenderLayer::RenderToCamera(const std::shared_ptr<Camera>& camera, const GlobalTransform& cameraModel)
{
	CameraInfoBlock cameraInfoBlock = {};
	camera->UpdateCameraInfoBlock(cameraInfoBlock, cameraModel);
	Graphics::UploadCameraInfo(cameraInfoBlock);
	auto deferredPrepassProgram = std::dynamic_pointer_cast<ShaderProgram>(Resources::GetResource("STANDARD_DEFERRED_PREPASS_PROGRAM"));
	auto deferredLightingProgram = std::dynamic_pointer_cast<ShaderProgram>(Resources::GetResource("STANDARD_DEFERRED_LIGHTING_PROGRAM"));
	auto& deferredRenderPass = m_renderPasses["DEFERRED_RENDERING"];
	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GlobalPipelineState& globalPipelineState) {
		VkExtent2D extent2D;
		const auto resolution = camera->GetSize();
		extent2D.width = resolution.x;
		extent2D.height = resolution.y;
		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = deferredRenderPass->GetVkRenderPass();
		renderPassBeginInfo.framebuffer = camera->GetFramebuffer()->GetVkFrameBuffer();
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = extent2D;

		VkClearValue clearColor = { {{1.0f, 0.0f, 0.0f, 0.0f}},  };

		std::vector<VkClearValue> clearValues = { 6, clearColor };

		renderPassBeginInfo.clearValueCount = clearValues.size();
		renderPassBeginInfo.pClearValues = clearValues.data();

		VkViewport viewport;
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(extent2D.width);
		viewport.height = static_cast<float>(extent2D.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent = extent2D;
		globalPipelineState.m_viewPort = viewport;
		globalPipelineState.m_scissor = scissor;

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		/*
		m_deferredInstancedRenderInstances[camera->GetHandle()].Dispatch([&](const std::shared_ptr<Material>& material)
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
					deferredPrepassProgram->BindAllDescriptorSets(commandBuffer);

					deferredPrepassProgram->BindShaders(commandBuffer);
					auto mesh = std::dynamic_pointer_cast<Mesh>(renderCommand.m_renderGeometry);
					mesh->DrawIndexed(commandBuffer, globalPipelineState);
					break;
				}
				}
			}
			);
		*/
		vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
		
		vkCmdEndRenderPass(commandBuffer);
		}
	);


	camera->m_rendered = true;
	camera->m_requireRendering = false;
}
