#include "PlayerController.hpp"

#include "Camera.hpp"
#include "Scene.hpp"
#include "EditorLayer.hpp"
#include "Time.hpp"
using namespace EvoEngine;

void PlayerController::OnCreate()
{
	m_startMouse = false;
}
void PlayerController::LateUpdate()
{
	auto scene = GetScene();

#pragma region Scene Camera Controller
	auto transform = scene->GetDataComponent<Transform>(GetOwner());
	const auto rotation = transform.GetRotation();
	auto position = transform.GetPosition();
	const auto front = rotation * glm::vec3(0, 0, -1);
	const auto right = rotation * glm::vec3(1, 0, 0);
	auto moved = false;
	if (scene->GetKey(GLFW_KEY_W) != KeyActionType::Release)
	{
		position += front * static_cast<float>(Time::DeltaTime()) * m_velocity;
		moved = true;
	}
	if (scene->GetKey(GLFW_KEY_S) != KeyActionType::Release)
	{
		position -= front * static_cast<float>(Time::DeltaTime()) * m_velocity;
		moved = true;
	}
	if (scene->GetKey(GLFW_KEY_A) != KeyActionType::Release)
	{
		position -= right * static_cast<float>(Time::DeltaTime()) * m_velocity;
		moved = true;
	}
	if (scene->GetKey(GLFW_KEY_D) != KeyActionType::Release)
	{
		position += right * static_cast<float>(Time::DeltaTime()) * m_velocity;
		moved = true;
	}
	if (scene->GetKey(GLFW_KEY_LEFT_SHIFT) != KeyActionType::Release)
	{
		position.y += m_velocity * static_cast<float>(Time::DeltaTime());
		moved = true;
	}
	if (scene->GetKey(GLFW_KEY_LEFT_CONTROL) != KeyActionType::Release)
	{
		position.y -= m_velocity * static_cast<float>(Time::DeltaTime());
		moved = true;
	}
	if (moved)
	{
		transform.SetPosition(position);
	}
	const glm::vec2 mousePosition = Input::GetMousePosition();
	float xOffset = 0;
	float yOffset = 0;
	if (mousePosition.x > FLT_MIN)
	{
		if (!m_startMouse)
		{
			m_lastX = mousePosition.x;
			m_lastY = mousePosition.y;
			m_startMouse = true;
		}
		xOffset = mousePosition.x - m_lastX;
		yOffset = -mousePosition.y + m_lastY;
		m_lastX = mousePosition.x;
		m_lastY = mousePosition.y;
	}
	if (scene->GetKey(GLFW_MOUSE_BUTTON_RIGHT) != KeyActionType::Release)
	{
		if (xOffset != 0 || yOffset != 0)
		{
			moved = true;
			m_sceneCameraYawAngle += xOffset * m_sensitivity;
			m_sceneCameraPitchAngle += yOffset * m_sensitivity;

			if (m_sceneCameraPitchAngle > 89.0f)
				m_sceneCameraPitchAngle = 89.0f;
			if (m_sceneCameraPitchAngle < -89.0f)
				m_sceneCameraPitchAngle = -89.0f;

			transform.SetRotation(
				Camera::ProcessMouseMovement(m_sceneCameraYawAngle, m_sceneCameraPitchAngle, false));
		}
	}
	if (moved)
	{
		scene->SetDataComponent(GetOwner(), transform);
	}
#pragma endregion
}

void PlayerController::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}
void PlayerController::Serialize(YAML::Emitter& out)
{
	out << YAML::Key << "m_velocity" << YAML::Value << m_velocity;
	out << YAML::Key << "m_sensitivity" << YAML::Value << m_sensitivity;
	out << YAML::Key << "m_sceneCameraYawAngle" << YAML::Value << m_sceneCameraYawAngle;
	out << YAML::Key << "m_sceneCameraPitchAngle" << YAML::Value << m_sceneCameraPitchAngle;
}
void PlayerController::Deserialize(const YAML::Node& in)
{
	if (in["m_velocity"]) m_velocity = in["m_velocity"].as<float>();
	if (in["m_sensitivity"]) m_sensitivity = in["m_sensitivity"].as<float>();
	if (in["m_sceneCameraYawAngle"]) m_sceneCameraYawAngle = in["m_sceneCameraYawAngle"].as<float>();
	if (in["m_sceneCameraPitchAngle"]) m_sceneCameraPitchAngle = in["m_sceneCameraPitchAngle"].as<float>();
}
void PlayerController::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	ImGui::DragFloat("Velocity", &m_velocity, 0.01f);
	ImGui::DragFloat("Mouse sensitivity", &m_sensitivity, 0.01f);
	ImGui::DragFloat("Yaw angle", &m_sceneCameraYawAngle, 0.01f);
	ImGui::DragFloat("Pitch angle", &m_sceneCameraPitchAngle, 0.01f);
}