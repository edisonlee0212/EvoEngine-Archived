#include "Application.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
using namespace EvoEngine;
int main() {

    Application::PushLayer<WindowLayer>();
    Application::PushLayer<RenderLayer>();

    Application::Initialize({});

    Application::Start();
    
	Application::Terminate();
    return 0;
}