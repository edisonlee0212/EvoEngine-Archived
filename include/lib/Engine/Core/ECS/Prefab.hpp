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

namespace EvoEngine
{
    struct AssimpNode
    {
        aiNode* m_correspondingNode = nullptr;
        std::string m_name;
        Transform m_localTransform;
        AssimpNode(aiNode* node);
        std::shared_ptr<AssimpNode> m_parent;
        std::vector<std::shared_ptr<AssimpNode>> m_children;
        std::shared_ptr<Bone> m_bone;
        bool m_hasMesh;

        bool NecessaryWalker(std::unordered_map<std::string, std::shared_ptr<Bone>>& boneMap);
        void AttachToAnimator(const std::shared_ptr<Animation>& animation, size_t& index) const;
        void AttachChild(const std::shared_ptr<Bone>& parent, size_t& index) const;
    };
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
        std::string m_name;
        bool m_enabled = true;
#pragma region Model Loading
        static void AttachAnimator(Prefab* parent, const Handle& animatorEntityHandle);

        static std::shared_ptr<Texture2D> CollectTexture(
            const std::string& directory,
            const std::string& path,
            std::unordered_map<std::string, std::shared_ptr<Texture2D>>& loadedTextures);

        static void ApplyBoneIndices(Prefab* node);
        static void ReadAnimations(
            const aiScene* importerScene,
            const std::shared_ptr<Animation>& animator,
            std::unordered_map<std::string, std::shared_ptr<Bone>>& bonesMap);
        static void ReadKeyFrame(BoneKeyFrames& boneAnimation, const aiNodeAnim* channel);
        static std::shared_ptr<Material> ReadMaterial(
            const std::string& directory,
            std::unordered_map<std::string, std::shared_ptr<Texture2D>>& texture2DsLoaded,
            const aiMaterial* importerMaterial);
        static bool ProcessNode(
            const std::string& directory,
            Prefab* modelNode,
            std::unordered_map<unsigned, std::shared_ptr<Material>>& loadedMaterials,
            std::unordered_map<std::string, std::shared_ptr<Texture2D>>& texture2DsLoaded,
            std::unordered_map<std::string, std::shared_ptr<Bone>>& bonesMap,
            const aiNode* importerNode,
            const std::shared_ptr<AssimpNode>& assimpNode,
            const aiScene* importerScene,
            const std::shared_ptr<Animation>& animation);
        static std::shared_ptr<Mesh> ReadMesh(aiMesh* importerMesh);
        static std::shared_ptr<SkinnedMesh> ReadSkinnedMesh(
            std::unordered_map<std::string, std::shared_ptr<Bone>>& bonesMap, aiMesh* importerMesh);
        void AttachChildren(const std::shared_ptr<Scene>& scene,
            const std::shared_ptr<Prefab>& modelNode,
            Entity parentEntity,
            const std::string& parentName,
            std::unordered_map<Handle, Handle>& map) const;

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
        void LoadModelInternal(const std::filesystem::path& path, bool optimize = false, unsigned flags = aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals);
        static void SaveModelInternal(const std::filesystem::path& path);


    public:
        void GatherAssets();

    	std::unordered_map<Handle, AssetRef> m_assets;

        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        Handle m_entityHandle = 0;
        std::vector<DataComponentHolder> m_dataComponents;
        std::vector<PrivateComponentHolder> m_privateComponents;
        std::vector<std::shared_ptr<Prefab>> m_children;
        template <typename T = IPrivateComponent> std::shared_ptr<T> GetPrivateComponent();
        void OnCreate() override;

        [[nodiscard]] Entity ToEntity(const std::shared_ptr<Scene>& scene) const;

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

} // namespace EvoEngine