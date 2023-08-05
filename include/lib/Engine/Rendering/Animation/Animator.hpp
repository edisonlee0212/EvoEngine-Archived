#pragma once
#include "Animation.hpp"
#include "Scene.hpp"
namespace EvoEngine
{
class Animator final : public IPrivateComponent
{
    std::vector<std::shared_ptr<Bone>> m_bones;
    friend class SkinnedMeshRenderer;
    friend class RenderLayer;

    std::vector<glm::mat4> m_transformChain;
    std::vector<glm::mat4> m_offsetMatrices;
    std::vector<std::string> m_names;
    AssetRef m_animation;
    size_t m_boneSize = 0;
    void BoneSetter(const std::shared_ptr<Bone> &boneWalker);
    void Setup();
    std::string m_currentActivatedAnimation;
    float m_currentAnimationTime;
    void Apply();
  public:
    /**
     * Only set offset matrices, so the animator can be used as ragDoll.
     * @param name Name of the bones
     * @param offsetMatrices The collection of offset matrices.
     */
    void Setup(const std::vector<std::string> &name, const std::vector<glm::mat4> &offsetMatrices);
    void ApplyOffsetMatrices();
    glm::mat4 GetReverseTransform(const int &index, const Entity &entity);

    [[nodiscard]] float GetCurrentAnimationTimePoint() const;
    [[nodiscard]] std::string GetCurrentAnimationName();
    void Animate(const std::string& animationName, float time);
    void Animate(float time);

    void OnDestroy() override;
    void Setup(const std::shared_ptr<Animation> &targetAnimation);
    void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

    void PostCloneAction(const std::shared_ptr<IPrivateComponent> &target) override;
    std::shared_ptr<Animation> GetAnimation();
    void Serialize(YAML::Emitter &out) override;
    void Deserialize(const YAML::Node &in) override;
    void CollectAssetRef(std::vector<AssetRef> &list) override;
};
} // namespace EvoEngine