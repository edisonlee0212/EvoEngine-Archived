#include "ProjectManager.hpp"

#include "Planet/PlanetTerrainSystem.hpp"
#include "Planet/TerrainChunk.hpp"

glm::dvec3 Planet::TerrainChunk::ChunkCenterPosition(glm::dvec3 planetPosition, double radius, glm::quat rotation)
{

    int actualDetailLevel = (int)glm::pow(2, m_detailLevel);
    glm::dvec2 percent = glm::dvec2(0.5, 0.5) / (double)actualDetailLevel;
    glm::dvec3 point = m_localUp +
                       (percent.x + (double)m_chunkCoordinate.x / (double)actualDetailLevel - 0.5) * 2 * m_axisA +
                       (percent.y + (double)m_chunkCoordinate.y / (double)actualDetailLevel - 0.5) * 2 * m_axisB;
    double x = rotation.x * 2.0f;
    double y = rotation.y * 2.0f;
    double z = rotation.z * 2.0f;
    double xx = rotation.x * x;
    double yy = rotation.y * y;
    double zz = rotation.z * z;
    double xy = rotation.x * y;
    double xz = rotation.x * z;
    double yz = rotation.y * z;
    double wx = rotation.w * x;
    double wy = rotation.w * y;
    double wz = rotation.w * z;

    glm::dvec3 res;
    res.x = (1.0f - (yy + zz)) * point.x + (xy - wz) * point.y + (xz + wy) * point.z;
    res.y = (xy + wz) * point.x + (1.0f - (xx + zz)) * point.y + (yz - wx) * point.z;
    res.z = (xz - wy) * point.x + (yz + wx) * point.y + (1.0f - (xx + yy)) * point.z;
    res = glm::normalize(res);
    glm::dvec3 ret = res * radius + planetPosition;
    return ret;
}

Planet::TerrainChunk::TerrainChunk(
    const std::shared_ptr<PlanetTerrain> &planetTerrain,
    const std::shared_ptr<TerrainChunk>& parent,
    unsigned detailLevel,
    glm::ivec2 chunkCoordinate,
    ChunkDirection direction,
    glm::dvec3 localUp)
{
    m_planetTerrain = planetTerrain;
    m_chunkCoordinate = chunkCoordinate;
    m_detailLevel = detailLevel;
    m_parent = parent;
    m_localUp = localUp;
    m_axisA = glm::dvec3(localUp.y, localUp.z, localUp.x);
    m_axisB = glm::cross(localUp, m_axisA);
    m_localUp = glm::normalize(m_localUp);
}

void Planet::TerrainChunk::Expand(std::mutex &mutex)
{
    if (!m_active)
        return;
    if (!m_c0)
    {
        auto chunk0 = std::make_shared<TerrainChunk>(
            m_planetTerrain.lock(),
            m_self.lock(),
            m_detailLevel + 1,
            glm::ivec2(m_chunkCoordinate.x * 2, m_chunkCoordinate.y * 2 + 1),
            ChunkDirection::UpperLeft,
            m_localUp);
        chunk0->m_self = chunk0;
        GenerateTerrain(mutex, chunk0);
        m_c0 = std::move(chunk0);
    }
    if (!m_c1)
    {
        auto chunk1 = std::make_shared<TerrainChunk>(
            m_planetTerrain.lock(),
            m_self.lock(),
            m_detailLevel + 1,
            glm::ivec2(m_chunkCoordinate.x * 2 + 1, m_chunkCoordinate.y * 2 + 1),
            ChunkDirection::UpperRight,
            m_localUp);
        chunk1->m_self = chunk1;
        GenerateTerrain(mutex, chunk1);
        m_c1 = std::move(chunk1);
    }
    if (!m_c2)
    {
        auto chunk2 = std::make_shared<TerrainChunk>(
            m_planetTerrain.lock(),
            m_self.lock(),
            m_detailLevel + 1,
            glm::ivec2(m_chunkCoordinate.x * 2, m_chunkCoordinate.y * 2),
            ChunkDirection::LowerLeft,
            m_localUp);
        chunk2->m_self = chunk2;
        GenerateTerrain(mutex, chunk2);
        m_c2 = std::move(chunk2);
    }
    if (!m_c3)
    {
        auto chunk3 = std::make_shared<TerrainChunk>(
            m_planetTerrain.lock(),
            m_self.lock(),
            m_detailLevel + 1,
            glm::ivec2(m_chunkCoordinate.x * 2 + 1, m_chunkCoordinate.y * 2),
            ChunkDirection::LowerRight,
            m_localUp);
        chunk3->m_self = chunk3;
        GenerateTerrain(mutex, chunk3);
        m_c3 = std::move(chunk3);
    }
    m_c0->m_active = true;
    m_c1->m_active = true;
    m_c2->m_active = true;
    m_c3->m_active = true;
    m_active = false;
    m_childrenActive = true;
}

void Planet::TerrainChunk::GenerateTerrain(std::mutex &mutex, std::shared_ptr<TerrainChunk> &targetChunk) const
{
    if (targetChunk->m_mesh)
    {
        Console::Error("Mesh Exist!");
    }
    auto planetTerrain = m_planetTerrain.lock();
    std::vector<Vertex> &vertices = planetTerrain->m_sharedVertices;
    auto size = vertices.size();
    auto resolution = planetTerrain->m_info.m_resolution;
    for (auto index = 0; index < size; index++)
    {
        int actualDetailLevel = (int)glm::pow(2, targetChunk->m_detailLevel);
        int x = index % resolution;
        int y = index / resolution;
        glm::dvec2 percent = glm::dvec2(x, y) / (double)(resolution - 1) / (double)actualDetailLevel;
        glm::dvec2 globalPercent =
            45.0 * glm::dvec2(
                       (percent.x + (double)targetChunk->m_chunkCoordinate.x / actualDetailLevel - 0.5) * 2.0,
                       (percent.y + (double)targetChunk->m_chunkCoordinate.y / actualDetailLevel - 0.5) * 2.0);
        glm::dvec2 actualPercent =
            glm::dvec2(glm::tan(glm::radians(globalPercent.x)), glm::tan(glm::radians(globalPercent.y)));
        glm::dvec3 pointOnUnitCube =
            targetChunk->m_localUp + actualPercent.x * targetChunk->m_axisA + actualPercent.y * targetChunk->m_axisB;
        pointOnUnitCube = glm::normalize(pointOnUnitCube);
        double elevation = 1.0;

        double previousResult = 1.0;
        for (const auto &stage : planetTerrain->m_terrainConstructionStages)
        {
            stage->Process(pointOnUnitCube, previousResult, elevation);
            previousResult = elevation;
        }
        vertices.at(index).m_position = glm::vec3(pointOnUnitCube * planetTerrain->m_info.m_radius * elevation);
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
    VertexAttributes attributes{};
    attributes.m_texCoord = true;

    mesh->SetVertices(attributes,
        planetTerrain->m_sharedVertices,
        planetTerrain->m_sharedTriangles);
    targetChunk->m_mesh = std::move(mesh);
}

void Planet::TerrainChunk::Collapse()
{
    if (!m_c0 || !m_c1 || !m_c2 || !m_c3)
        return;
    if (!m_c0->m_active || !m_c1->m_active || !m_c2->m_active || !m_c3->m_active)
        return;

    m_c0->m_active = false;
    m_c1->m_active = false;
    m_c2->m_active = false;
    m_c3->m_active = false;
    m_active = true;
    m_childrenActive = false;

    m_c0.reset();
    m_c1.reset();
    m_c2.reset();
    m_c3.reset();
}
