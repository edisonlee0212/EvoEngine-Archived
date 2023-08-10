#include "Application.hpp"

#include "AnimationPlayer.hpp"
#include "Prefab.hpp"
#include "ClassRegistry.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "Scene.hpp"
#include "Time.hpp"
#include "ProjectManager.hpp"
#include "WindowLayer.hpp"
#include "EditorLayer.hpp"
#include "Jobs.hpp"
#include "TransformGraph.hpp"
#include "Input.hpp"
#include "Lights.hpp"
#include "Mesh.hpp"
#include "MeshRenderer.hpp"
#include "Resources.hpp"
#include "Shader.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "Cubemap.hpp"
#include "EnvironmentalMap.hpp"
#include "LightProbe.hpp"
#include "PlayerController.hpp"
#include "ReflectionProbe.hpp"
#include "RigidBody.hpp"
#include "PhysicsLayer.hpp"
#include "Joint.hpp"
#include "Particles.hpp"
#include "PostProcessingStack.hpp"
#include "UnknownPrivateComponent.hpp"
using namespace EvoEngine;

void Application::PreUpdateInternal()
{
	const auto& application = GetInstance();
	const auto now = std::chrono::system_clock::now();
	const std::chrono::duration<double> deltaTime = now - Time::m_lastUpdateTime;
	Time::m_deltaTime = deltaTime.count();
	Time::m_lastUpdateTime = std::chrono::system_clock::now();
	if (application.m_applicationStatus == ApplicationStatus::Uninitialized)
	{
		EVOENGINE_ERROR("Application uninitialized!")
			return;
	}
	if (application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	Input::PreUpdate();
	Graphics::PreUpdate();

	if (application.m_applicationStatus == ApplicationStatus::NoProject) return;
	TransformGraph::CalculateTransformGraphs(application.m_activeScene);
	for (const auto& i : application.m_externalPreUpdateFunctions)
		i();
	if (application.m_applicationStatus == ApplicationStatus::Playing || application.m_applicationStatus == ApplicationStatus::Step)
	{
		application.m_activeScene->Start();
	}
	for (const auto& i : application.m_layers)
	{
		i->PreUpdate();
	}
	if (Time::m_steps == 0) {
		Time::m_lastFixedUpdateTime = std::chrono::system_clock::now();
		Time::m_steps = 1;
	}
	const auto lastFixedUpdateTime = Time::m_lastFixedUpdateTime;
	std::chrono::duration<double> duration = std::chrono::system_clock::now() - lastFixedUpdateTime;
	size_t step = 1;
	while (duration.count() >= step * Time::m_timeStep)
	{
		for (const auto& i : application.m_externalFixedUpdateFunctions)
			i();
		for (const auto& i : application.m_layers)
		{
			i->FixedUpdate();
		}
		if (application.m_applicationStatus == ApplicationStatus::Playing || application.m_applicationStatus == ApplicationStatus::Step)
		{
			application.m_activeScene->FixedUpdate();
		}
		duration = std::chrono::system_clock::now() - lastFixedUpdateTime;
		step++;
		const auto now = std::chrono::system_clock::now();
		const std::chrono::duration<double> fixedDeltaTime = now - Time::m_lastFixedUpdateTime;
		Time::m_fixedDeltaTime = fixedDeltaTime.count();
		Time::m_lastFixedUpdateTime = std::chrono::system_clock::now();
		if (step > 10)
		{
			EVOENGINE_WARNING("Fixed update timeout!");
		} break;
	}
}

void Application::UpdateInternal()
{
	const auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::Uninitialized)
	{
		EVOENGINE_ERROR("Application uninitialized!")
			return;
	}
	if (application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	if (application.m_applicationStatus == ApplicationStatus::NoProject) return;

	for (const auto& i : application.m_externalUpdateFunctions)
		i();

	for (auto& i : application.m_layers)
	{
		i->Update();
	}
	if (application.m_applicationStatus == ApplicationStatus::Playing || application.m_applicationStatus == ApplicationStatus::Step)
	{
		application.m_activeScene->Update();
	}
}

void Application::LateUpdateInternal()
{
	auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::Uninitialized)
	{
		EVOENGINE_ERROR("Application uninitialized!")
			return;
	}
	if (application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	if (application.m_applicationStatus == ApplicationStatus::NoProject)
	{
		const auto editorLayer = GetLayer<EditorLayer>();
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();
		const auto windowLayer = GetLayer<WindowLayer>();
		if (windowLayer && ImGui::BeginMainMenuBar())
		{
			FileUtils::SaveFile(
				"Create or load New Project",
				"Project",
				{ ".eveproj" },
				[&](const std::filesystem::path& path) {
					ProjectManager::GetOrCreateProject(path);
					if (ProjectManager::GetInstance().m_projectFolder)
					{
						windowLayer->ResizeWindow(
							application.m_applicationInfo.m_defaultWindowSize.x,
							application.m_applicationInfo.m_defaultWindowSize.y);
						application.m_applicationStatus = ApplicationStatus::Stop;
					}
				},
				false);
			ImGui::EndMainMenuBar();
		}

		Graphics::AppendCommands([&](const VkCommandBuffer commandBuffer)
			{
				Graphics::EverythingBarrier(commandBuffer);
				constexpr VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
				VkRect2D renderArea;
				renderArea.offset = { 0, 0 };
				renderArea.extent = Graphics::GetSwapchain()->GetImageExtent();
				Graphics::TransitImageLayout(commandBuffer,
					Graphics::GetSwapchain()->GetVkImage(), Graphics::GetSwapchain()->GetImageFormat(), 1,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

				VkRenderingAttachmentInfo colorAttachmentInfo{};
				colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
				colorAttachmentInfo.imageView = Graphics::GetSwapchain()->GetVkImageView();
				colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
				colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
				colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				colorAttachmentInfo.clearValue = clearColor;

				VkRenderingInfo renderInfo{};
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.colorAttachmentCount = 1;
				renderInfo.pColorAttachments = &colorAttachmentInfo;

				vkCmdBeginRendering(commandBuffer, &renderInfo);

				ImGui::Render();
				ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

				vkCmdEndRendering(commandBuffer);
				Graphics::TransitImageLayout(commandBuffer,
					Graphics::GetSwapchain()->GetVkImage(), Graphics::GetSwapchain()->GetImageFormat(), 1,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
			});

	}
	else {
		for (const auto& i : application.m_externalLateUpdateFunctions)
			i();

		if (application.m_applicationStatus == ApplicationStatus::Playing || application.m_applicationStatus == ApplicationStatus::Step)
		{
			application.m_activeScene->LateUpdate();
		}
		for (auto i = application.m_layers.rbegin(); i != application.m_layers.rend(); ++i)
		{
			(*i)->LateUpdate();
		}
		if (application.m_applicationStatus == ApplicationStatus::Step)
			application.m_applicationStatus = ApplicationStatus::Pause;
	}
	Graphics::LateUpdate();
}

const ApplicationInfo& Application::GetApplicationInfo()
{
	auto& application = GetInstance();
	return application.m_applicationInfo;
}

const ApplicationStatus& Application::GetApplicationStatus()
{
	const auto& application = GetInstance();
	return application.m_applicationStatus;
}

std::shared_ptr<Scene> Application::GetActiveScene()
{
	auto& application = GetInstance();
	return application.m_activeScene;
}

void Application::Reset()
{
	auto& application = GetInstance();
	application.m_applicationStatus = ApplicationStatus::Stop;
	Time::m_steps = Time::m_frames = 0;
}

void Application::Initialize(const ApplicationInfo& applicationCreateInfo)
{
	auto& application = GetInstance();
	if (application.m_applicationStatus != ApplicationStatus::Uninitialized) {
		EVOENGINE_ERROR("Application is not uninitialzed!")
			return;
	}
	application.m_applicationInfo = applicationCreateInfo;
	InitializeRegistry();
	Jobs::Initialize();
	Entities::Initialize();
	TransformGraph::Initialize();
	Graphics::Initialize();
	Resources::Initialize();
	Graphics::PostResourceLoadingInitialization();
	Resources::InitializeEnvironmentalMap();

	for (const auto& layer : application.m_layers)
	{
		layer->OnCreate();
	}
	const auto windowLayer = GetLayer<WindowLayer>();
	const auto editorLayer = GetLayer<EditorLayer>();
	if (!application.m_applicationInfo.m_projectPath.empty())
	{
		ProjectManager::GetOrCreateProject(application.m_applicationInfo.m_projectPath);
		if (ProjectManager::GetInstance().m_projectFolder)
		{
			if (windowLayer) {
				windowLayer->ResizeWindow(
					application.m_applicationInfo.m_defaultWindowSize.x,
					application.m_applicationInfo.m_defaultWindowSize.y);
			}
			application.m_applicationStatus = ApplicationStatus::Stop;
		}
	}
	else if (!windowLayer || !editorLayer) {
		throw std::runtime_error("Project must present when there's no EditorLayer or WindowLayer!");
	}
	else
	{
		application.m_applicationStatus = ApplicationStatus::NoProject;
	}
}

void Application::Start()
{
	const auto& application = GetInstance();

	Time::m_startTime = std::chrono::system_clock::now();
	Time::m_steps = Time::m_frames = 0;
	const auto editorLayer = GetLayer<EditorLayer>();
	if(!editorLayer) Play();
	while (application.m_applicationStatus != ApplicationStatus::OnDestroy)
	{
		PreUpdateInternal();
		UpdateInternal();
		LateUpdateInternal();
	}
}

void Application::End()
{
	GetInstance().m_applicationStatus = ApplicationStatus::OnDestroy;
}

void Application::Terminate()
{
	const auto& application = GetInstance();
	for (auto i = application.m_layers.rbegin(); i != application.m_layers.rend(); ++i)
	{
		(*i)->OnDestroy();
	}
}

const std::vector<std::shared_ptr<ILayer>>& Application::GetLayers()
{
	const auto& application = GetInstance();
	return application.m_layers;
}

void Application::Attach(const std::shared_ptr<Scene>& scene)
{
	auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::Playing)
	{
		EVOENGINE_ERROR("Stop Application to attach scene");
	}

	application.m_activeScene = scene;
	for (auto& func : application.m_postAttachSceneFunctions)
	{
		func(scene);
	}
	for (const auto& layer : application.m_layers)
	{
		layer->m_scene = scene;
	}
}

void Application::Play()
{
	auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::NoProject || application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	if (application.m_applicationStatus != ApplicationStatus::Pause && application.m_applicationStatus != ApplicationStatus::Stop)
		return;
	if (application.m_applicationStatus == ApplicationStatus::Stop)
	{
		const auto copiedScene = ProjectManager::CreateTemporaryAsset<Scene>();
		Scene::Clone(ProjectManager::GetStartScene().lock(), copiedScene);
		Attach(copiedScene);
	}
	application.m_applicationStatus = ApplicationStatus::Playing;
}
void Application::Stop()
{
	auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::NoProject || application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	if (application.m_applicationStatus == ApplicationStatus::Stop) return;
	application.m_applicationStatus = ApplicationStatus::Stop;
	Attach(ProjectManager::GetStartScene().lock());
}
void Application::Pause()
{
	auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::NoProject || application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	if (application.m_applicationStatus != ApplicationStatus::Playing)
		return;
	application.m_applicationStatus = ApplicationStatus::Pause;
}

void Application::Step()
{
	auto& application = GetInstance();
	if (application.m_applicationStatus != ApplicationStatus::Pause && application.m_applicationStatus != ApplicationStatus::Stop)
		return;
	if (application.m_applicationStatus == ApplicationStatus::Stop)
	{
		const auto copiedScene = ProjectManager::CreateTemporaryAsset<Scene>();
		Scene::Clone(ProjectManager::GetStartScene().lock(), copiedScene);
		Attach(copiedScene);
	}
	application.m_applicationStatus = ApplicationStatus::Step;
}

void Application::InitializeRegistry()
{
	ClassRegistry::RegisterDataComponent<Ray>("Ray");

	ClassRegistry::RegisterPrivateComponent<Joint>("Joint");
	ClassRegistry::RegisterPrivateComponent<RigidBody>("RigidBody");
	ClassRegistry::RegisterPrivateComponent<Camera>("Camera");
	ClassRegistry::RegisterPrivateComponent<AnimationPlayer>("AnimationPlayer");
	ClassRegistry::RegisterPrivateComponent<PlayerController>("PlayerController");
	ClassRegistry::RegisterPrivateComponent<Particles>("Particles");
	ClassRegistry::RegisterPrivateComponent<MeshRenderer>("MeshRenderer");
	
	ClassRegistry::RegisterPrivateComponent<SkinnedMeshRenderer>("SkinnedMeshRenderer");
	ClassRegistry::RegisterPrivateComponent<Animator>("Animator");
	ClassRegistry::RegisterPrivateComponent<PointLight>("PointLight");
	ClassRegistry::RegisterPrivateComponent<SpotLight>("SpotLight");
	ClassRegistry::RegisterPrivateComponent<DirectionalLight>("DirectionalLight");
	ClassRegistry::RegisterPrivateComponent<UnknownPrivateComponent>("UnknownPrivateComponent");

	ClassRegistry::RegisterSystem<PhysicsSystem>("PhysicsSystem");

	ClassRegistry::RegisterAsset<PostProcessingStack>("PostProcessingStack", { ".evepostprocessingstack" });
	ClassRegistry::RegisterAsset<IAsset>("IAsset", { ".eveasset" });
	ClassRegistry::RegisterAsset<Material>("Material", { ".evematerial" });
	ClassRegistry::RegisterAsset<Collider>("Collider", { ".uecollider" });
	ClassRegistry::RegisterAsset<Cubemap>("Cubemap", { ".evecubemap" });
	ClassRegistry::RegisterAsset<LightProbe>("LightProbe", { ".evelightprobe" });
	ClassRegistry::RegisterAsset<ReflectionProbe>("ReflectionProbe", { ".evereflectionprobe" });
	ClassRegistry::RegisterAsset<EnvironmentalMap>("EnvironmentalMap", { ".eveenvironmentalmap" });
	ClassRegistry::RegisterAsset<Shader>("Shader", { ".eveshader" });
	ClassRegistry::RegisterAsset<Mesh>("Mesh", { ".evemesh" });

	ClassRegistry::RegisterAsset<Prefab>("Prefab", { ".eveprefab", ".obj", ".gltf", ".glb", ".blend", ".ply", ".fbx", ".dae", ".x3d" });
	ClassRegistry::RegisterAsset<Texture2D>("Texture2D", { ".png", ".jpg", ".jpeg", ".tga", ".hdr" });
	ClassRegistry::RegisterAsset<Scene>("Scene", { ".evescene" });
	ClassRegistry::RegisterAsset<ParticleInfoList>("ParticleInfoList", { ".eveparticleinfolist" });
	ClassRegistry::RegisterAsset<Animation>("Animation", { ".eveanimation" });
	ClassRegistry::RegisterAsset<SkinnedMesh>("SkinnedMesh", { ".eveskinnedmesh" });
	ClassRegistry::RegisterAsset<PhysicsMaterial>("PhysicsMaterial", { ".evephysicsmaterial" });
}

void Application::RegisterPreUpdateFunction(const std::function<void()>& func)
{
	GetInstance().m_externalPreUpdateFunctions.push_back(func);
}

void Application::RegisterUpdateFunction(const std::function<void()>& func)
{
	GetInstance().m_externalUpdateFunctions.push_back(func);
}

void Application::RegisterLateUpdateFunction(const std::function<void()>& func)
{
	GetInstance().m_externalLateUpdateFunctions.push_back(func);
}
void Application::RegisterFixedUpdateFunction(const std::function<void()>& func)
{
	GetInstance().m_externalFixedUpdateFunctions.push_back(func);
}

void Application::RegisterPostAttachSceneFunction(
	const std::function<void(const std::shared_ptr<Scene>& newScene)>& func)
{
	GetInstance().m_postAttachSceneFunctions.push_back(func);
}

bool Application::IsPlaying()
{
	const auto& application = GetInstance();
	return application.m_applicationStatus == ApplicationStatus::Playing;
}
