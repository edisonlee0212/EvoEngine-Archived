#include "Input.hpp"
#include "Scene.hpp"
#include "Application.hpp"
#include "WindowLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;

void Input::KeyCallBack(GLFWwindow* window, int key, int scanCode, int action, int mods)
{
	auto& input = GetInstance();
	if(action == GLFW_PRESS)
	{
		if(const auto search = input.m_pressedKeys.find(key); search != input.m_pressedKeys.end())
		{
			//Dispatch hold if the key is already pressed.
			search->second = KeyActionType::Hold;
			Dispatch({ key, KeyActionType::Hold });
		}else
		{
			//Dispatch press if the key is previously released.
			input.m_pressedKeys.insert({ key, KeyActionType::Press });
			Dispatch({ key, KeyActionType::Press });
		}
		//std::cout << "P" + std::to_string(key) << std::endl;
	}else if(action == GLFW_RELEASE)
	{
		if (input.m_pressedKeys.find(key) != input.m_pressedKeys.end())
		{
			//Dispatch hold if the key is already pressed.
			input.m_pressedKeys.erase(key);
			Dispatch({ key, KeyActionType::Release });
		}
		//std::cout << "R" + std::to_string(key) << std::endl;
	}
}

void Input::MouseButtonCallBack(GLFWwindow* window, int button, int action, int mods)
{
	auto& input = GetInstance();
	if (action == GLFW_PRESS)
	{
		if (const auto search = input.m_pressedKeys.find(button); search != input.m_pressedKeys.end())
		{
			//Dispatch hold if the key is already pressed.
			search->second = KeyActionType::Hold;
			Dispatch({ button, KeyActionType::Hold });
		}
		else
		{
			//Dispatch press if the key is previously released.
			input.m_pressedKeys.insert({ button, KeyActionType::Press });
			Dispatch({ button, KeyActionType::Press });
		}
		//std::cout << "P" + std::to_string(button) << std::endl;
	}
	else if (action == GLFW_RELEASE)
	{
		if (input.m_pressedKeys.find(button) != input.m_pressedKeys.end())
		{
			//Dispatch hold if the key is already pressed.
			input.m_pressedKeys.erase(button);
			Dispatch({ button, KeyActionType::Release });
		}
		//std::cout << "R" + std::to_string(button) << std::endl;
	}
}

void Input::Dispatch(const InputEvent& event)
{
	const auto &layers = Application::GetLayers();
	if(!layers.empty())
	{
		layers[0]->OnInputEvent(event);
	}
	if (!Application::GetLayer<EditorLayer>()) {
		const auto activeScene = Application::GetActiveScene();
		
		auto& scenePressedKeys = activeScene->m_pressedKeys;
		if (event.m_keyAction == KeyActionType::Press)
		{
			if (const auto search = scenePressedKeys.find(event.m_key); search != activeScene->m_pressedKeys.end())
			{
				//Dispatch hold if the key is already pressed.
				search->second = KeyActionType::Hold;
			}
			else
			{
				//Dispatch press if the key is previously released.
				scenePressedKeys.insert({ event.m_key, KeyActionType::Press });
			}
		}
		else if (event.m_keyAction == KeyActionType::Release)
		{
			if (scenePressedKeys.find(event.m_key) != scenePressedKeys.end())
			{
				//Dispatch hold if the key is already pressed.
				scenePressedKeys.erase(event.m_key);
			}
		}
	}
}

void Input::PreUpdate()
{
	auto& input = GetInstance();
	input.m_mousePosition = { FLT_MIN, FLT_MIN };

	if (const auto windowLayer = Application::GetLayer<WindowLayer>())
	{
		glfwPollEvents();
		double x = FLT_MIN;
		double y = FLT_MIN;
		glfwGetCursorPos(windowLayer->GetGlfwWindow(), &x, &y);
		input.m_mousePosition = { x, y };
	}
}

glm::vec2 Input::GetMousePosition()
{
	const auto& input = GetInstance();
	return input.m_mousePosition;
}

KeyActionType Input::GetKey(const int key)
{
	const auto& input = GetInstance();
	if (const auto search = input.m_pressedKeys.find(key); search != input.m_pressedKeys.end()) 
		return search->second;
	return KeyActionType::Release;
}
