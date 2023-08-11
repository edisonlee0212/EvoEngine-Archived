#pragma once
#include "Material.hpp"
#include "Strands.hpp"
#include "IPrivateComponent.hpp"
namespace EvoEngine {
    class StrandsRenderer : public IPrivateComponent {
        void RenderBound(const std::shared_ptr<EditorLayer>& editorLayer, glm::vec4& color);
    public:
        bool m_castShadow = true;
        AssetRef m_strands;
        AssetRef m_material;
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void OnCreate() override;
        void Serialize(YAML::Emitter& out) override;
        void Deserialize(const YAML::Node& in) override;
        void OnDestroy() override;
        void CollectAssetRef(std::vector<AssetRef>& list) override;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };
}
