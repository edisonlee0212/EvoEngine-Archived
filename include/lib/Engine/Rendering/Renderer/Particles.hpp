#pragma once
#include "Material.hpp"
#include "Mesh.hpp"
#include "Scene.hpp"
namespace EvoEngine
{
    class Particles : public IPrivateComponent
    {
    public:
        void OnCreate() override;
        Bound m_boundingBox;
        bool m_castShadow = true;
        AssetRef m_particleInfoList;
        AssetRef m_mesh;
        AssetRef m_material;
        void RecalculateBoundingBox();
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void Serialize(YAML::Emitter& out) override;
        void Deserialize(const YAML::Node& in) override;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
        void CollectAssetRef(std::vector<AssetRef>& list) override;
        void OnDestroy() override;
    };
} // namespace EvoEngine
