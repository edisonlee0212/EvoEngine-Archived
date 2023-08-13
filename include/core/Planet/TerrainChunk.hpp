#pragma once
#include "Application.hpp"
#include "Mesh.hpp"
using namespace EvoEngine;
namespace Planet
{
enum class ChunkDirection
{
    Root,
    UpperLeft,
    UpperRight,
    LowerLeft,
    LowerRight
};
class PlanetTerrain;
class TerrainChunk
{
    std::weak_ptr<PlanetTerrain> m_planetTerrain;
    std::weak_ptr<TerrainChunk> m_self;
  public:
    std::shared_ptr<Mesh> m_mesh;
    // The level of detail, the larger the detail, the smaller the chunk will be.
    unsigned m_detailLevel;
    // The chunk coordinate in which a chunk belongs to the face
    glm::ivec2 m_chunkCoordinate;
    ChunkDirection m_direction;
    std::weak_ptr<TerrainChunk> m_parent;

    bool m_childrenActive = false;
    bool m_active = false;
    // The index of four children, upperleft = 0, upperright = 1, lower left = 2, lower right = 3.
    std::shared_ptr<TerrainChunk> m_c0;
    std::shared_ptr<TerrainChunk> m_c1;
    std::shared_ptr<TerrainChunk> m_c2;
    std::shared_ptr<TerrainChunk> m_c3;
    glm::dvec3 m_localUp;
    glm::dvec3 m_axisA;
    glm::dvec3 m_axisB;
    glm::dvec3 ChunkCenterPosition(glm::dvec3 planetPosition, double radius, glm::quat rotation);
    TerrainChunk(
        const std::shared_ptr<PlanetTerrain>& planetTerrain,
        const std::shared_ptr<TerrainChunk>& parent,
        unsigned detailLevel,
        glm::ivec2 chunkCoordinate,
        ChunkDirection direction,
        glm::dvec3 localUp);
    void Expand(std::mutex &mutex);
    void GenerateTerrain(std::mutex &mutex, std::shared_ptr<TerrainChunk> &targetChunk) const;
    void Collapse();
};
} // namespace Planet
