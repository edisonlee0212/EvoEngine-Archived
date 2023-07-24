#pragma once
#include "IPrivateComponent.hpp"

namespace EvoEngine
{
    class MeshRenderer : public IPrivateComponent
    {
        friend class Editor;
        void RenderBound(glm::vec4& color);
    public:
        bool m_forwardRendering = false;
        bool m_castShadow = true;
        bool m_receiveShadow = true;
        AssetRef m_mesh;
        AssetRef m_material;
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void Serialize(YAML::Emitter& out) override;
        void Deserialize(const YAML::Node& in) override;
        void OnDestroy() override;
        void CollectAssetRef(std::vector<AssetRef>& list) override;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };
}
