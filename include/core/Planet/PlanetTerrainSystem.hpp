#pragma once
#include "Application.hpp"
#include "Planet/PlanetTerrain.hpp"
#include "Camera.hpp"
#include "Material.hpp"
#include "ISystem.hpp"
using namespace EvoEngine;
namespace Planet
{
class PlanetTerrainSystem : public ISystem
{
    friend class PlanetTerrain;
  public:
    void OnCreate() override;
    void Update() override;
    void FixedUpdate() override;
    void CheckLod(
        std::mutex &mutex,
        std::shared_ptr<TerrainChunk> &chunk,
        const PlanetInfo &info,
        const GlobalTransform &planetTransform,
        const GlobalTransform &cameraTransform);
    void RenderChunk(
        std::shared_ptr<TerrainChunk> &chunk,
        const std::shared_ptr<Material> &material,
        glm::mat4 &matrix,
        bool receiveShadow) const;
};
} // namespace Planet