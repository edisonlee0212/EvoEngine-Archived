#pragma once
#include <unordered_set>

#include "ISingleton.hpp"

namespace evo_engine
{
	enum class KeyActionType
	{
		Press,
		Hold,
		Release,
		Unknown
	};
	struct InputEvent
	{
		int m_key = GLFW_KEY_UNKNOWN;
		KeyActionType m_keyAction = KeyActionType::Unknown;
	};
	class Input : public ISingleton<Input>
	{
		friend class Graphics;
		friend class Application;
		friend class EditorLayer;
		std::unordered_map<int, KeyActionType> m_pressedKeys = {};
		glm::vec2 m_mousePosition = glm::vec2(0.0f);

		static void KeyCallBack(GLFWwindow* window, int key, int scanCode, int action, int mods);
		static void MouseButtonCallBack(GLFWwindow* window, int button, int action, int mods);
		static void Dispatch(const InputEvent &event);
		static void PreUpdate();

		static KeyActionType GetKey(int key);
	public:
		static glm::vec2 GetMousePosition();
	};
}
