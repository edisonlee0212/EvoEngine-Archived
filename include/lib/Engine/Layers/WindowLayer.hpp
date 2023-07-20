#pragma once
#include "ILayer.hpp"
namespace EvoEngine
{
	class WindowLayer final : public ILayer
	{
		friend class Graphics;
		friend class RenderLayer;
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

	public:
		GLFWwindow* GetGlfwWindow() const;
		void ResizeWindow(int x, int y) const;
		bool GetKey(int key) const;

		bool GetMouseButton(int button) const;
		[[nodiscard]] glm::vec2 GetMousePosition() const;
	};
}