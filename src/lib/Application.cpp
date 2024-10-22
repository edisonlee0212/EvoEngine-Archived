#include "Application.hpp"

#include "AnimationPlayer.hpp"
#include "Prefab.hpp"
#include "ClassRegistry.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "Scene.hpp"
#include "Times.hpp"
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
#include "LODGroup.hpp"
#include "PlayerController.hpp"
#include "ReflectionProbe.hpp"
#include "Particles.hpp"
#include "PointCloud.hpp"
#include "PostProcessingStack.hpp"
#include "RenderLayer.hpp"
#include "StrandsRenderer.hpp"
#include "UnknownPrivateComponent.hpp"
#include "Strands.hpp"
#include "WayPoints.hpp"
using namespace EvoEngine;

void Application::PreUpdateInternal()
{
	auto& application = GetInstance();
	const auto now = std::chrono::system_clock::now();
	const std::chrono::duration<double> deltaTime = now - Times::m_lastUpdateTime;
	Times::m_deltaTime = deltaTime.count();
	Times::m_lastUpdateTime = std::chrono::system_clock::now();
	if (application.m_applicationStatus == ApplicationStatus::Uninitialized)
	{
		EVOENGINE_ERROR("Application uninitialized!")
			return;
	}
	if (application.m_applicationStatus == ApplicationStatus::OnDestroy) return;

	application.m_applicationExecutionStatus = ApplicationExecutionStatus::PreUpdate;
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
	if (Times::m_steps == 0) {
		Times::m_lastFixedUpdateTime = std::chrono::system_clock::now();
		Times::m_steps = 1;
	}
	const auto lastFixedUpdateTime = Times::m_lastFixedUpdateTime;
	std::chrono::duration<double> duration = std::chrono::system_clock::now() - lastFixedUpdateTime;
	size_t step = 1;
	while (duration.count() >= step * Times::m_timeStep)
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
		const std::chrono::duration<double> fixedDeltaTime = now - Times::m_lastFixedUpdateTime;
		Times::m_fixedDeltaTime = fixedDeltaTime.count();
		Times::m_lastFixedUpdateTime = std::chrono::system_clock::now();
		if (step > 10)
		{
			EVOENGINE_WARNING("Fixed update timeout!");
		} break;
	}

	if (const auto renderLayer = GetLayer<RenderLayer>())
	{
		renderLayer->ClearAllCameras();
	}
}

void Application::UpdateInternal()
{
	auto& application = GetInstance();
	if (application.m_applicationStatus == ApplicationStatus::Uninitialized)
	{
		EVOENGINE_ERROR("Application uninitialized!")
			return;
	}
	if (application.m_applicationStatus == ApplicationStatus::OnDestroy) return;
	if (application.m_applicationStatus == ApplicationStatus::NoProject) return;
	application.m_applicationExecutionStatus = ApplicationExecutionStatus::Update;
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

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();
		const auto windowLayer = GetLayer<WindowLayer>();
		if (windowLayer)
		{
			ImGuiFileDialog::Instance()->OpenDialog("ChooseProjectKey", "Choose Project", ".eveproj", ".");
#pragma region Dock
			static bool opt_fullscreen_persistant = true;
			bool opt_fullscreen = opt_fullscreen_persistant;
			static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

			// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
			// because it would be confusing to have two docking targets within each others.
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
			if (opt_fullscreen)
			{
				ImGuiViewport* viewport = ImGui::GetMainViewport();
				ImGui::SetNextWindowPos(viewport->WorkPos);
				ImGui::SetNextWindowSize(viewport->WorkSize);
				ImGui::SetNextWindowViewport(viewport->ID);
				ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
				ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
				window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
					ImGuiWindowFlags_NoMove;
				window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
			}

			// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
			// and handle the pass-thru hole, so we ask Begin() to not render a background.
			if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
				window_flags |= ImGuiWindowFlags_NoBackground;

			// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
			// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
			// all active windows docked into it will lose their parent and become undocked.
			// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
			// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
			static bool openDock = true;
			ImGui::Begin("Root DockSpace", &openDock, window_flags);
			ImGui::PopStyleVar();
			if (opt_fullscreen)
				ImGui::PopStyleVar(2);
			ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
			ImGui::End();
#pragma endregion
			ImGui::SetNextWindowDockID(dockspace_id);
			// display
			if (ImGuiFileDialog::Instance()->Display("ChooseProjectKey"))
			{
				// action if OK
				if (ImGuiFileDialog::Instance()->IsOk())
				{
					// action
					std::filesystem::path path = ImGuiFileDialog::Instance()->GetFilePathName();
					ProjectManager::GetOrCreateProject(path);
					if (ProjectManager::GetInstance().m_projectFolder)
					{
						windowLayer->ResizeWindow(
							application.m_applicationInfo.m_defaultWindowSize.x,
							application.m_applicationInfo.m_defaultWindowSize.y);
						application.m_applicationStatus = ApplicationStatus::Stop;
					}
				}
				// close
				ImGuiFileDialog::Instance()->Close();
			}
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
		if (const auto editorLayer = GetLayer<EditorLayer>())
		{
			for (const auto& layer : application.m_layers) layer->OnInspect(editorLayer);
		}
		application.m_applicationExecutionStatus = ApplicationExecutionStatus::LateUpdate;
		for (const auto& i : application.m_externalLateUpdateFunctions)
			i();
		if (const auto renderLayer = GetLayer<RenderLayer>())
		{
			renderLayer->RenderAllCameras();
		}
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
	Times::m_steps = Times::m_frames = 0;
}

void Application::Initialize(const ApplicationInfo& applicationCreateInfo)
{
	auto& application = GetInstance();

	if (application.m_applicationStatus != ApplicationStatus::Uninitialized) {
		EVOENGINE_ERROR("Application is not uninitialzed!")
			return;
	}
	application.m_applicationInfo = applicationCreateInfo;
	const auto windowLayer = GetLayer<WindowLayer>();
	const auto editorLayer = GetLayer<EditorLayer>();
	if (!application.m_applicationInfo.m_projectPath.empty()) {
		if (application.m_applicationInfo.m_projectPath.extension().string() != ".eveproj")
		{
			EVOENGINE_ERROR("Project file extension is not eveproj!");
			return;
		}
	}
	else if (!windowLayer || !editorLayer)
	{
		EVOENGINE_ERROR("Project filepath must present when there's no EditorLayer or WindowLayer!");
		return;
	}
	const auto defaultThreadSize = std::thread::hardware_concurrency();
	Jobs::Initialize(defaultThreadSize - 2);
	InitializeRegistry();
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
	else
	{
		application.m_applicationStatus = ApplicationStatus::NoProject;
		if (windowLayer) {
			windowLayer->ResizeWindow(800, 600);
		}
	}
}

void Application::Start()
{
	Times::m_startTime = std::chrono::system_clock::now();
	Times::m_steps = Times::m_frames = 0;
	if (const auto editorLayer = GetLayer<EditorLayer>(); !editorLayer) Play();

}

void Application::Run()
{
	while (Loop());
}

bool Application::Loop()
{
	const auto& application = GetInstance();
	if (application.m_applicationStatus != ApplicationStatus::OnDestroy) {
		PreUpdateInternal();
		UpdateInternal();
		LateUpdateInternal();
		return true;
	}
	return false;
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

	ClassRegistry::RegisterPrivateComponent<Camera>("Camera");
	ClassRegistry::RegisterPrivateComponent<AnimationPlayer>("AnimationPlayer");
	ClassRegistry::RegisterPrivateComponent<PlayerController>("PlayerController");
	ClassRegistry::RegisterPrivateComponent<Particles>("Particles");
	ClassRegistry::RegisterPrivateComponent<MeshRenderer>("MeshRenderer");
	ClassRegistry::RegisterPrivateComponent<StrandsRenderer>("StrandsRenderer");
	ClassRegistry::RegisterPrivateComponent<SkinnedMeshRenderer>("SkinnedMeshRenderer");
	ClassRegistry::RegisterPrivateComponent<Animator>("Animator");
	ClassRegistry::RegisterPrivateComponent<PointLight>("PointLight");
	ClassRegistry::RegisterPrivateComponent<SpotLight>("SpotLight");
	ClassRegistry::RegisterPrivateComponent<DirectionalLight>("DirectionalLight");
	ClassRegistry::RegisterPrivateComponent<WayPoints>("WayPoints");
	ClassRegistry::RegisterPrivateComponent<LodGroup>("LodGroup");
	ClassRegistry::RegisterPrivateComponent<UnknownPrivateComponent>("UnknownPrivateComponent");

	ClassRegistry::RegisterAsset<PostProcessingStack>("PostProcessingStack", { ".evepostprocessingstack" });
	ClassRegistry::RegisterAsset<IAsset>("IAsset", { ".eveasset" });
	ClassRegistry::RegisterAsset<Material>("Material", { ".evematerial" });

	ClassRegistry::RegisterAsset<Cubemap>("Cubemap", { ".evecubemap" });
	ClassRegistry::RegisterAsset<LightProbe>("LightProbe", { ".evelightprobe" });
	ClassRegistry::RegisterAsset<ReflectionProbe>("ReflectionProbe", { ".evereflectionprobe" });
	ClassRegistry::RegisterAsset<EnvironmentalMap>("EnvironmentalMap", { ".eveenvironmentalmap" });
	ClassRegistry::RegisterAsset<Shader>("Shader", { ".eveshader" });
	ClassRegistry::RegisterAsset<Mesh>("Mesh", { ".evemesh" });
	ClassRegistry::RegisterAsset<Strands>("Strands", { ".evestrands", ".hair" });
	ClassRegistry::RegisterAsset<Prefab>("Prefab",
		{ ".eveprefab", ".obj", ".gltf", ".glb", ".blend", ".ply", ".fbx", ".dae", ".x3d" });
	ClassRegistry::RegisterAsset<Texture2D>("Texture2D", { ".evetexture2d", ".png", ".jpg", ".jpeg", ".tga", ".hdr" });
	ClassRegistry::RegisterAsset<Scene>("Scene", { ".evescene" });
	ClassRegistry::RegisterAsset<ParticleInfoList>("ParticleInfoList", { ".eveparticleinfolist" });
	ClassRegistry::RegisterAsset<Animation>("Animation", { ".eveanimation" });
	ClassRegistry::RegisterAsset<SkinnedMesh>("SkinnedMesh", { ".eveskinnedmesh" });

	ClassRegistry::RegisterAsset<PointCloud>("PointCloud", { ".evepointcloud" });
}

ApplicationExecutionStatus Application::GetApplicationExecutionStatus()
{
	return GetInstance().m_applicationExecutionStatus;
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
