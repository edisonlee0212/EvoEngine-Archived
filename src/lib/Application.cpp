#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "Scene.hpp"
using namespace EvoEngine;


void ApplicationTime::OnInspect()
{
    if (ImGui::CollapsingHeader("Time Settings"))
    {
        float timeStep = m_timeStep;
        if (ImGui::DragFloat("Time step", &timeStep, 0.001f, 0.001f, 1.0f))
        {
            m_timeStep = timeStep;
        }
    }
}
double ApplicationTime::TimeStep() const
{
    return m_timeStep;
}
void ApplicationTime::SetTimeStep(const double& value)
{
    m_timeStep = value;
}
double ApplicationTime::FixedDeltaTime() const
{
    return glfwGetTime() - m_lastFixedUpdateTime;
}

double ApplicationTime::DeltaTime() const
{
    return m_deltaTime;
}
double ApplicationTime::CurrentTime() const
{
    return glfwGetTime();
}

double ApplicationTime::LastFrameTime() const
{
    return m_lastUpdateTime;
}
void ApplicationTime::StartFixedUpdate()
{
    m_fixedUpdateTimeStamp = glfwGetTime();
}

void ApplicationTime::EndFixedUpdate()
{
    m_lastFixedUpdateTime = m_fixedUpdateTimeStamp;
}

void Application::PreUpdateInternal()
{
    auto& application = GetInstance();
    glfwPollEvents();
    if (glfwWindowShouldClose(Graphics::GetGlfwWindow()))
    {
        application.m_applicationStatus = ApplicationStatus::OnDestroy;
    }
    application.m_time.m_deltaTime = glfwGetTime() - application.m_time.m_frameStartTime;
    application.m_time.m_frameStartTime = glfwGetTime();
    //Editor::ImGuiPreUpdate();
    //OpenGLUtils::PreUpdate();
    Graphics::DrawFrame();
    if (application.m_applicationStatus == ApplicationStatus::Initialized)
    {
        //Inputs::PreUpdate();
        for (const auto& i : application.m_externalPreUpdateFunctions)
            i();

        if (application.m_gameStatus == GameStatus::Playing || application.m_gameStatus == GameStatus::Step)
        {
            application.m_activeScene->Start();
        }
        for (auto& i : application.m_layers)
        {
            i->PreUpdate();
        }
        auto fixedDeltaTime = application.m_time.FixedDeltaTime();
        if (fixedDeltaTime >= application.m_time.m_timeStep)
        {
            application.m_time.StartFixedUpdate();
            for (const auto& i : application.m_externalFixedUpdateFunctions)
                i();
            for (auto& i : application.m_layers)
            {
                i->FixedUpdate();
            }
            if (application.m_gameStatus == GameStatus::Playing || application.m_gameStatus == GameStatus::Step)
            {
                application.m_activeScene->FixedUpdate();
            }
            application.m_time.EndFixedUpdate();
        }
    }
}

void Application::UpdateInternal()
{
    auto& application = GetInstance();
    if (application.m_applicationStatus == ApplicationStatus::Initialized)
    {
        for (const auto& i : application.m_externalUpdateFunctions)
            i();

        for (auto& i : application.m_layers)
        {
            i->Update();
        }
        if (application.m_gameStatus == GameStatus::Playing || application.m_gameStatus == GameStatus::Step)
        {
            application.m_activeScene->Update();
        }
    }
}

void Application::LateUpdateInternal()
{
    auto& application = GetInstance();
    if (application.m_applicationStatus == ApplicationStatus::Initialized)
    {
        for (const auto& i : application.m_externalLateUpdateFunctions)
            i();

        if (application.m_gameStatus == GameStatus::Playing || application.m_gameStatus == GameStatus::Step)
        {
            application.m_activeScene->LateUpdate();
        }

        for (auto& i : application.m_layers)
        {
            i->LateUpdate();
        }
        for (auto& i : application.m_layers)
        {
            i->OnInspect();
        }
        // Post-processing happens here
        // Manager settings
        //OnInspect();
        if (application.m_gameStatus == GameStatus::Step)
            application.m_gameStatus = GameStatus::Pause;
    }
    /*
    else
    {
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
    }*/
    // ImGui drawing
    //Editor::ImGuiLateUpdate();
    // Swap Window's framebuffer
    //Windows::LateUpdate();
    application.m_time.m_lastUpdateTime = glfwGetTime();
}

std::shared_ptr<Scene> Application::GetActiveScene()
{
    auto& application = GetInstance();
    return application.m_activeScene;
}

void Application::Initialize(const ApplicationCreateInfo& applicationCreateInfo)
{
	auto& application = GetInstance();
	if (application.m_initialized) return;
	application.m_name = applicationCreateInfo.m_applicationName;

#pragma region Graphics
	VkApplicationInfo vkApplicationInfo{};
	vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	vkApplicationInfo.pApplicationName = applicationCreateInfo.m_applicationName.c_str();
	vkApplicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	vkApplicationInfo.pEngineName = "EvoEngine";
	vkApplicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	vkApplicationInfo.apiVersion = VK_API_VERSION_1_0;
	Graphics::Initialize(applicationCreateInfo, vkApplicationInfo);
#pragma endregion
}

void Application::Start()
{
    auto& application = GetInstance();
    /*
    for (auto& i : application.m_layers)
    {
        i->OnCreate();
    }
    application.m_applicationStatus = ApplicationStatus::Uninitialized;
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
    application.m_gameStatus = GameStatus::Stop;
    */
    while (application.m_applicationStatus != ApplicationStatus::OnDestroy)
    {
        PreUpdateInternal();
        UpdateInternal();
        LateUpdateInternal();
    }
}

void Application::Terminate()
{
	auto& application = GetInstance();
	Graphics::Terminate();
	
}
