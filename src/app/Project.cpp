#include "Application.hpp"
#include <shaderc/shaderc.hpp>
using namespace EvoEngine;
int main() {
    Application::Initialize({});

    Application::Start();
    
	Application::Terminate();
    return 0;
}