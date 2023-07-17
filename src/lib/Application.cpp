#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "Scene.hpp"
#include "Time.hpp"
#include "ProjectManager.hpp"
using namespace EvoEngine;

void Application::PreUpdateInternal()
{
	const auto& application = GetInstance();
	Time::m_lastUpdateTime = std::chrono::system_clock::now();
	if (application.m_applicationStatus == ApplicationStatus::Uninitialized)
	{
		EVOENGINE_ERROR("Application uninitialized!")
			return;
	}
	if (application.m_applicationStatus == ApplicationStatus::OnDestroy) return;

	for (const auto& i : application.m_externalPreUpdateFunctions)
		i();
	Graphics::PreUpdate();
	if (application.m_applicationStatus == ApplicationStatus::Playing || application.m_applicationStatus == ApplicationStatus::Step)
	{
		application.m_activeScene->Start();
	}
	for (const auto& i : application.m_layers)
	{
		i->PreUpdate();
	}
	if (Time::m_steps == 0) Time::m_lastFixedUpdateTime = std::chrono::system_clock::now();
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
		Time::m_lastFixedUpdateTime = std::chrono::system_clock::now();
		if (step > 10)
		{
			EVOENGINE_WARNING("Fixed update timeout!");
			break;
		}
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

	for (const auto& i : application.m_externalUpdateFunctions)
		i();

	Graphics::Update();
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
		/*
		if (ImGui::BeginMainMenuBar())
		{
			FileUtils::SaveFile(
				"Create or load New Project",
				"Project",
				{ ".ueproj" },
				[&](const std::filesystem::path& path) {
					ProjectManager::GetOrCreateProject(path);
					if (ProjectManager::GetInstance().m_projectFolder)
					{
						Windows::ResizeWindow(
							application.m_applicationConfigs.m_defaultWindowSize.x,
							application.m_applicationConfigs.m_defaultWindowSize.y);
						application.m_applicationStatus = ApplicationStatus::Initialized;
					}
				},
				false);
			ImGui::EndMainMenuBar();
		}
		*/

	}
	else {
		for (const auto& i : application.m_externalLateUpdateFunctions)
			i();

		if (application.m_applicationStatus == ApplicationStatus::Playing || application.m_applicationStatus == ApplicationStatus::Step)
		{
			application.m_activeScene->LateUpdate();
		}
	}
	for (const auto& i : application.m_layers)
	{
		i->LateUpdate();
	}
	for (const auto& i : application.m_layers)
	{
		i->OnInspect();
	}
	// Post-processing happens here
	// Manager settings
	//OnInspect();
	Graphics::LateUpdate();
	if (application.m_applicationStatus == ApplicationStatus::Step)
		application.m_applicationStatus = ApplicationStatus::Pause;
	// ImGui drawing
	//Editor::ImGuiLateUpdate();
}

const ApplicationInfo& Application::GetApplicationInfo()
{
	return GetInstance().m_applicationInfo;
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

	Graphics::Initialize();
	for (const auto& layer : application.m_layers)
	{
		layer->OnCreate();
	}
	application.m_applicationStatus = ApplicationStatus::NoProject;
}

void Application::Start()
{
	auto& application = GetInstance();
	/*
	if (!application.m_applicationConfigs.m_projectPath.empty())
	{
		ProjectManager::GetOrCreateProject(application.m_applicationConfigs.m_projectPath);
		if (ProjectManager::GetInstance().m_projectFolder)
		{
			Windows::ResizeWindow(
				application.m_applicationConfigs.m_defaultWindowSize.x,
				application.m_applicationConfigs.m_defaultWindowSize.y);
			application.m_applicationStatus = ApplicationStatus::Initialized;
		}
	}
	*/
	Time::m_startTime = std::chrono::system_clock::now();
	Time::m_steps = Time::m_frames = 0;
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
	for (auto& layer : application.m_layers)
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
		auto copiedScene = ProjectManager::CreateTemporaryAsset<Scene>();
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
