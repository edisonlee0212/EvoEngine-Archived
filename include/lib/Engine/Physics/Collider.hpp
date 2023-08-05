#pragma once
#include "Entities.hpp"
#include "IAsset.hpp"
#include <PhysicsMaterial.hpp>

namespace EvoEngine
{
enum class ShapeType
{
    Sphere,
    Box,
    Capsule
};
class Collider : public IAsset
{
    friend class PhysicsLayer;
    friend class RigidBody;
    PxShape *m_shape = nullptr;
    glm::vec3 m_shapeParam = glm::vec3(1.0f);
    ShapeType m_shapeType = ShapeType::Box;
    AssetRef m_physicsMaterial;

    size_t m_attachCount = 0;

  public:
    void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
    void OnCreate() override;
    ~Collider();
    void SetShapeType(const ShapeType& type);
    void SetShapeParam(const glm::vec3& param);
    void SetMaterial(const std::shared_ptr<PhysicsMaterial>& material);
    void CollectAssetRef(std::vector<AssetRef> &list) override;
    void Serialize(YAML::Emitter &out) override;
    void Deserialize(const YAML::Node &in) override;
};
} // namespace EvoEngine