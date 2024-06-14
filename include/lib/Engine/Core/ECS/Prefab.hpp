#pragma once
#include <Animator.hpp>
#include <IAsset.hpp>
#include <IPrivateComponent.hpp>
#include <ISystem.hpp>
#include <Material.hpp>
#include <Mesh.hpp>
#include <SkinnedMesh.hpp>
#include <Transform.hpp>

#include "Texture2D.hpp"

namespace evo_engine
{
   
    struct DataComponentHolder
    {
        DataComponentType m_type;
        std::shared_ptr<IDataComponent> m_data;

        void Serialize(YAML::Emitter& out) const;
        bool Deserialize(const YAML::Node& in);
    };

    struct PrivateComponentHolder
    {
        bool m_enabled;
        std::shared_ptr<IPrivateComponent> m_data;

        void Serialize(YAML::Emitter& out) const;
        void Deserialize(const YAML::Node& in);
    };

    class Prefab : public IAsset
    {
       
        bool m_enabled = true;
#pragma region Model Loading
        static void AttachAnimator(Prefab* parent, const Handle& animatorEntityHandle);
        static void ApplyBoneIndices(Prefab* node);
        static void AttachChildren(const std::shared_ptr<Scene>& scene,
                                   const std::shared_ptr<Prefab>& modelNode,
                                   Entity parentEntity,
                                   std::unordered_map<Handle, Handle>& map);

        void AttachChildrenPrivateComponent(const std::shared_ptr<Scene>& scene,
            const std::shared_ptr<Prefab>& modelNode,
            const Entity& parentEntity,
            const std::unordered_map<Handle, Handle>& map) const;
        static void RelinkChildren(const std::shared_ptr<Scene>& scene, const Entity& parentEntity, const std::unordered_map<Handle, Handle>& map);
#pragma endregion

        static bool OnInspectComponents(const std::shared_ptr<Prefab>& walker);
        static bool OnInspectWalker(const std::shared_ptr<Prefab>& walker);
        static void GatherAssetsWalker(const std::shared_ptr<Prefab>& walker, std::unordered_map<Handle, AssetRef>& assets);
    protected:
        bool LoadInternal(const std::filesystem::path& path) override;
        bool SaveInternal(const std::filesystem::path& path) const override;
        bool LoadModelInternal(const std::filesystem::path& path, bool optimize = false, unsigned flags = aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals);
        bool SaveModelInternal(const std::filesystem::path& path) const;


    public:
    	std::string m_name;
        void GatherAssets();

    	std::unordered_map<Handle, AssetRef> m_assets;

        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        Handle m_entityHandle = Handle();
        std::vector<DataComponentHolder> m_dataComponents;
        std::vector<PrivateComponentHolder> m_privateComponents;
        std::vector<std::shared_ptr<Prefab>> m_children;
        template <typename T = IPrivateComponent> std::shared_ptr<T> GetPrivateComponent();
        void OnCreate() override;

        [[maybe_unused]] Entity ToEntity(const std::shared_ptr<Scene>& scene, bool autoAdjustSize = false) const;

        void LoadModel(const std::filesystem::path& path, bool optimize = false, unsigned flags = aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals);

        void FromEntity(const Entity& entity);
        void CollectAssets(std::unordered_map<Handle, std::shared_ptr<IAsset>>& map) const;
        void Serialize(YAML::Emitter& out) const override;
        void Deserialize(const YAML::Node& in) override;
    };

    template <typename T> std::shared_ptr<T> Prefab::GetPrivateComponent()
    {
        auto typeName = Serialization::GetSerializableTypeName<T>();
        for (auto& i : m_privateComponents)
        {
            if (i.m_data->GetTypeName() == typeName)
            {
                return std::static_pointer_cast<T>(i.m_data);
            }
        }
        return nullptr;
    }

} // namespace evo_engine