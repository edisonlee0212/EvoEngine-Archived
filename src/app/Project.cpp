#include "AnimationLayer.hpp"
#include "Application.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;
int main() {

    Application::PushLayer<WindowLayer>();
    Application::PushLayer<EditorLayer>();
    //Application::PushLayer<RenderLayer>();
    //Application::PushLayer<AnimationLayer>();

    ApplicationInfo applicationInfo;
    const std::filesystem::path resourceFolderPath("../../../Resources");
    applicationInfo.m_projectPath = resourceFolderPath / "Example Projects/Test/1.ueproj";
    Application::Initialize(applicationInfo);

    Application::Start();
    
	Application::Terminate();
    return 0;
}