#include "pybind11/pybind11.h"
#include "AnimationPlayer.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "PlayerController.hpp"
#include "Prefab.hpp"
#include "Time.hpp"
#include "PhysicsLayer.hpp"
#include "PostProcessingStack.hpp"
using namespace EvoEngine;
namespace py = pybind11;

void StartEngine() {
    Application::PushLayer<WindowLayer>();
	Application::PushLayer<PhysicsLayer>();
    Application::PushLayer<EditorLayer>();
    Application::PushLayer<RenderLayer>();

    ApplicationInfo applicationInfo;
    Application::Initialize(applicationInfo);
    Application::Start();

    Application::Terminate();
}

void RenderDefaultScene(const std::string &projectPath, const int resolutionX, const int resolutionY, const std::string& outputPath)
{
    if (!std::filesystem::is_regular_file(projectPath))
    {
        EVOENGINE_ERROR("Project doesn't exist!");
        return;
    }
    if (resolutionX <= 0 || resolutionY <= 0)
    {
        EVOENGINE_ERROR("Resolution error!");
        return;
    }
    Application::PushLayer<PhysicsLayer>();
    Application::PushLayer<RenderLayer>();
    
    ApplicationInfo applicationInfo;
    applicationInfo.m_projectPath = std::filesystem::path(projectPath);
    ProjectManager::SetScenePostLoadActions([&](const std::shared_ptr<Scene>& scene)
    {
            const auto mainCamera = scene->m_mainCamera.Get<Camera>();
            mainCamera->Resize({ resolutionX, resolutionY });
    });
	Application::Initialize(applicationInfo);
    EVOENGINE_LOG("Loaded scene at " + projectPath);
	Application::Start(false);
	Application::Loop();
	Graphics::WaitForDeviceIdle();
	auto scene = Application::GetActiveScene();
	const auto mainCamera = scene->m_mainCamera.Get<Camera>();
	mainCamera->GetRenderTexture()->StoreToPng(outputPath);
    EVOENGINE_LOG("Exported image to " + outputPath);
	Application::Terminate();
}

PYBIND11_MODULE(PyEvoEngine, m) {
    m.doc() = "EvoEngine"; // optional module docstring
    m.def("Start", &StartEngine, "Start EvoEngine");

	m.def("Render", &RenderDefaultScene, "Render Target Project Default Scene");
}