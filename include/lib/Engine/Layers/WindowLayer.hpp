#pragma once
#include "ILayer.hpp"
namespace EvoEngine
{
	class WindowLayer final : public ILayer
	{
		friend class Graphics;
#pragma region Presenters
		std::vector<GLFWmonitor*> m_monitors;
		GLFWmonitor* m_primaryMonitor = nullptr;
		GLFWwindow* m_window = nullptr;
		glm::ivec2 m_windowSize = { 1 , 1 };
#pragma endregion

		static void FramebufferResizeCallback(GLFWwindow*, int, int);
		static void SetMonitorCallback(GLFWmonitor* monitor, int event);
		static void WindowFocusCallback(GLFWwindow* window, int focused);

		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void LateUpdate() override;
		void OnInspect() override;
	public:
		GLFWwindow* GetGlfwWindow() const;

		
	};
}