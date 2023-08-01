#include "AnimationLayer.hpp"
#include "Application.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;
int main() {

    Application::PushLayer<WindowLayer>();
    //Application::PushLayer<EditorLayer>();
    Application::PushLayer<RenderLayer>();
    //Application::PushLayer<AnimationLayer>();

    ApplicationInfo applicationInfo;
    const std::filesystem::path resourceFolderPath("../../../Resources");
    applicationInfo.m_projectPath = resourceFolderPath / "Example Projects/Test/1.ueproj";
    Application::Initialize(applicationInfo);
    std::shared_ptr<Texture2D> texture2D = std::dynamic_pointer_cast<Texture2D>(ProjectManager::GetOrCreateAsset("border.png"));

    auto scene = Application::GetActiveScene();
    auto mainCameraEntity = scene->CreateEntity("MainCamera");
	auto mainCamera =  scene->GetOrSetPrivateComponent<Camera>(mainCameraEntity).lock();
    scene->m_mainCamera = mainCamera;
    mainCamera->m_clearColor = { 0, 0, 0 };
    Application::Start();
    
	Application::Terminate();
    return 0;
}