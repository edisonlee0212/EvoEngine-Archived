#include "Application.hpp"
#include <shaderc/shaderc.hpp>

#include "WindowLayer.hpp"
#include "GraphicsLayer.hpp"
using namespace EvoEngine;
int main() {

    Application::PushLayer<WindowLayer>();
    Application::PushLayer<GraphicsLayer>();


    Application::Initialize({});

    Application::Start();
    
	Application::Terminate();
    return 0;
}