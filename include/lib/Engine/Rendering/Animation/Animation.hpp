#pragma once
#include "Scene.hpp"
#include "Transform.hpp"
namespace evo_engine
{
#pragma region Bone
struct BonePosition
{
    glm::vec3 m_value;
    float m_timeStamp;
};

struct BoneRotation
{
    glm::quat m_value;
    float m_timeStamp;
};

struct BoneScale
{
    glm::vec3 m_value;
    float m_timeStamp;
};

struct BoneKeyFrames
{
    std::vector<BonePosition> m_positions;
    std::vector<BoneRotation> m_rotations;
    std::vector<BoneScale> m_scales;
    float m_maxTimeStamp = 0.0f;
    /* Gets the current index on mKeyPositions to interpolate to based on the current
    animation time */
    int GetPositionIndex(const float &animationTime);

    /* Gets the current index on mKeyRotations to interpolate to based on the current
    animation time */
    int GetRotationIndex(const float &animationTime);

    /* Gets the current index on mKeyScalings to interpolate to based on the current
    animation time */
    int GetScaleIndex(const float &animationTime);

    /* Gets normalized value for Lerp & Slerp*/
    static float GetScaleFactor(const float &lastTimeStamp, const float &nextTimeStamp, const float &animationTime);

    /* figures out which position keys to interpolate b/w and performs the interpolation
    and returns the translation matrix */
    glm::mat4 InterpolatePosition(const float &animationTime);

    /* figures out which rotations keys to interpolate b/w and performs the interpolation
    and returns the rotation matrix */
    glm::mat4 InterpolateRotation(const float &animationTime);

    /* figures out which scaling keys to interpolate b/w and performs the interpolation
    and returns the scale matrix */
    glm::mat4 InterpolateScaling(const float &animationTime);

    void Serialize(YAML::Emitter &out) const;
    void Deserialize(const YAML::Node &in);
};

struct Bone
{
    std::map<std::string, BoneKeyFrames> m_animations;
    std::string m_name;
    Transform m_offsetMatrix = Transform();
    size_t m_index;
    std::vector<std::shared_ptr<Bone>> m_children;
    /* Interpolates b/w positions,rotations & scaling keys based on the current time of the
    animation and prepares the local transformation matrix by combining all keys transformations */
    void Animate(
        const std::string &name,
        const float &animationTime,
        const glm::mat4 &parentTransform,
        const glm::mat4 &rootTransform,
        std::vector<glm::mat4> &results);
    bool OnInspect();

    void Serialize(YAML::Emitter &out) const;
    void Deserialize(const YAML::Node &in);
};

#pragma endregion
class Animation : public IAsset
{
  public:
    std::map<std::string, float> m_animationLength;
    std::shared_ptr<Bone> m_rootBone;
    size_t m_boneSize = 0;
    [[nodiscard]] std::shared_ptr<Bone>& UnsafeGetRootBone();
    [[nodiscard]] std::map<std::string, float>&UnsafeGetAnimationLengths();
    bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
    void Animate(
        const std::string &name,
        const float &animationTime,
        const glm::mat4 &rootTransform,
        std::vector<glm::mat4> &results);
    [[nodiscard]] std::string GetFirstAvailableAnimationName();
    [[nodiscard]] float GetAnimationLength(const std::string& animationName) const;
    [[nodiscard]] bool HasAnimation(const std::string& animationName) const;
    [[nodiscard]] bool IsEmpty() const;
    void Serialize(YAML::Emitter &out) const override;
    void Deserialize(const YAML::Node &in) override;
};
} // namespace evo_engine