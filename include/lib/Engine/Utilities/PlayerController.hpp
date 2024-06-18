#pragma once
#include "IPrivateComponent.hpp"

namespace EvoEngine
{
    class PlayerController : public IPrivateComponent
    {
        float m_lastX = 0, m_lastY = 0, m_lastScrollY = 0;
        bool m_startMouse = false;
        float m_sceneCameraYawAngle = -89;
        float m_sceneCameraPitchAngle = 0;
    public:
        float m_velocity = 20.0f;
        float m_sensitivity = 0.1f;
        void OnCreate() override;
        void LateUpdate() override;
        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void Serialize(YAML::Emitter& out) const override;
        void Deserialize(const YAML::Node& in) override;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };
}
