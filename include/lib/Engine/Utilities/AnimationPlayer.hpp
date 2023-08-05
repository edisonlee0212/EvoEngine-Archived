#pragma once
#include "IPrivateComponent.hpp"

namespace EvoEngine
{
    class AnimationPlayer : public IPrivateComponent
    {
    public:
        bool m_autoPlay = true;
        float m_autoPlaySpeed = 30.0f;
        void Update() override;
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        //void Serialize(YAML::Emitter& out) override;
        //void Deserialize(const YAML::Node& in) override;
    };
}
