#pragma once
#include "IPrivateComponent.hpp"

namespace EvoEngine
{
    class MeshRenderer : public IPrivateComponent
    {
        
        void RenderBound(const std::shared_ptr<EditorLayer>& editorLayer, const glm::vec4& color);
    public:
        bool m_castShadow = true;
        AssetRef m_mesh;
        AssetRef m_material;
        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void Serialize(YAML::Emitter& out) const override;
        void Deserialize(const YAML::Node& in) override;
        void OnDestroy() override;
        void CollectAssetRef(std::vector<AssetRef>& list) override;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };
}
