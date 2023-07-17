#include "Application.hpp"
#include <shaderc/shaderc.hpp>

#include "WindowLayer.hpp"
#include "Graphics.hpp"
using namespace EvoEngine;
int main() {

    Application::PushLayer<WindowLayer>();


    Application::Initialize({});

    Application::Start();
    
	Application::Terminate();
    return 0;
}