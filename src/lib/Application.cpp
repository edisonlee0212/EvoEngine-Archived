#include "Application.hpp"
#include "Window.hpp"
#include "Graphics.hpp"
using namespace EvoEngine;



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

void Application::Terminate()
{
	auto& application = GetInstance();
	Graphics::Terminate();
	
}
