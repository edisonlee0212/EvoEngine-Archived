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

    Application::Initialize({});

    Application::Start();
    
	Application::Terminate();
    return 0;
}