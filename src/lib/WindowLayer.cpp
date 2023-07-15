#include "WindowLayer.hpp"

#include "Application.hpp"

using namespace EvoEngine;

void WindowLayer::WindowResizeCallback(GLFWwindow* window, int width, int height)
{
	auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer->m_window == window)
	{
		windowLayer->m_windowSize = { width, height };
	}

}

void WindowLayer::SetMonitorCallback(GLFWmonitor* monitor, int event)
{
	auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (event == GLFW_CONNECTED)
	{
		// The monitor was connected
		for (const auto& i : windowLayer->m_monitors)
			if (i == monitor)
				return;
		windowLayer->m_monitors.push_back(monitor);
	}
	else if (event == GLFW_DISCONNECTED)
	{
		// The monitor was disconnected
		for (auto i = 0; i < windowLayer->m_monitors.size(); i++)
		{
			if (monitor == windowLayer->m_monitors[i])
			{
				windowLayer->m_monitors.erase(windowLayer->m_monitors.begin() + i);
			}
		}
	}
	windowLayer->m_primaryMonitor = glfwGetPrimaryMonitor();

}

void WindowLayer::WindowFocusCallback(GLFWwindow* window, int focused)
{
	auto& windowLayer = Application::GetLayer<WindowLayer>();
	/*
	if (focused)
	{
		ProjectManager::ScanProject();
	}
	 */
}

void WindowLayer::OnCreate()
{
#pragma region Windows
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	int size;
	const auto monitors = glfwGetMonitors(&size);
	for (auto i = 0; i < size; i++)
	{
		m_monitors.push_back(monitors[i]);
	}
	m_primaryMonitor = glfwGetPrimaryMonitor();
	glfwSetMonitorCallback(SetMonitorCallback);
	const auto& applicationInfo = Application::GetApplicationInfo();
	m_windowSize = applicationInfo.m_defaultWindowSize;
	m_window = glfwCreateWindow(m_windowSize.x, m_windowSize.y, applicationInfo.m_applicationName.c_str(), nullptr, nullptr);

	if (applicationInfo.m_fullScreen)
		glfwMaximizeWindow(m_window);
	glfwSetFramebufferSizeCallback(m_window, WindowResizeCallback);
	glfwSetWindowFocusCallback(m_window, WindowFocusCallback);
	if (m_window == nullptr)
	{
		EVOENGINE_ERROR("Failed to create a window");
	}
#pragma endregion
}

void WindowLayer::OnDestroy()
{
#pragma region Windows
	glfwDestroyWindow(m_window);
	glfwTerminate();
#pragma endregion
}

void WindowLayer::PreUpdate()
{
	glfwPollEvents();
	if (glfwWindowShouldClose(m_window))
	{
		Application::End();
	}
}

void WindowLayer::LateUpdate()
{
}

void WindowLayer::OnInspect()
{
}

GLFWwindow* WindowLayer::GetGlfwWindow() const
{
	return m_window;
}
