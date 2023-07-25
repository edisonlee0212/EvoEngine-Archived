#pragma once
#include "Animator.hpp"
#include "Scene.hpp"
#include "Vertex.hpp"
#include "GraphicsResources.hpp"
namespace EvoEngine
{
class SkinnedMesh : public IAsset
{
    std::unique_ptr<Buffer> m_verticesBuffer = {};
    std::unique_ptr<Buffer> m_trianglesBuffer = {};

    size_t m_offset = 0;

    unsigned m_mask = 0;
    Bound m_bound;
    friend class SkinnedMeshRenderer;
    friend class Particles;
    friend class Graphics;
    friend class RenderLayer;
    
    size_t m_version = 0;
    std::vector<SkinnedVertex> m_skinnedVertices;
    std::vector<glm::uvec3> m_triangles;
    friend struct SkinnedMeshBonesBlock;
    
    //Don't serialize.
    std::vector<std::shared_ptr<Bone>> m_bones;
    friend class Prefab;
  protected:
    bool SaveInternal(const std::filesystem::path &path) override;
  public:
    void OnCreate() override;
    void FetchIndices();

    //Need serialize
    std::vector<unsigned> m_boneAnimatorIndices;
    static void TryInitialize();
    static void UploadBones(const std::vector<glm::mat4> &matrices);
    void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
    [[nodiscard]] glm::vec3 GetCenter() const;
    [[nodiscard]] Bound GetBound() const;
    void Upload();
    void SetVertices(const unsigned &mask, std::vector<SkinnedVertex> &skinnedVertices, std::vector<unsigned> &indices);
    void SetVertices(const unsigned &mask, std::vector<SkinnedVertex> &skinnedVertices, std::vector<glm::uvec3> &triangles);
    [[nodiscard]] size_t GetSkinnedVerticesAmount() const;
    [[nodiscard]] size_t GetTriangleAmount() const;
    void RecalculateNormal();
    void RecalculateTangent();
    [[nodiscard]] size_t &GetVersion();
    [[nodiscard]] std::vector<SkinnedVertex> &UnsafeGetSkinnedVertices();
    [[nodiscard]] std::vector<glm::uvec3> &UnsafeGetTriangles();

    void Serialize(YAML::Emitter &out) override;
    void Deserialize(const YAML::Node &in) override;
};
} // namespace EvoEngine