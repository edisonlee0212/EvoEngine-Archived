#include "Planet/PlanetTerrainSystem.hpp"
#include "ClassRegistry.hpp"
#include "Scene.hpp"
#include "RenderLayer.hpp"
using namespace Planet;

void PlanetTerrainSystem::Update()
{
    auto scene = GetScene();
    const std::vector<Entity> *const planetTerrainList =
        scene->UnsafeGetPrivateComponentOwnersList<PlanetTerrain>();
    if (planetTerrainList == nullptr)
        return;

    std::mutex meshGenLock;
    const auto mainCamera = scene->m_mainCamera.Get<Camera>();
    if (mainCamera)
    {
        const auto cameraLtw = scene->GetDataComponent<GlobalTransform>(mainCamera->GetOwner());
        for (auto i = 0; i < planetTerrainList->size(); i++)
        {
            auto planetTerrain = scene->GetOrSetPrivateComponent<PlanetTerrain>(planetTerrainList->at(i)).lock();
            if (!planetTerrain->IsEnabled())
                continue;
            auto &planetInfo = planetTerrain->m_info;
            auto planetTransform = scene->GetDataComponent<GlobalTransform>(planetTerrain->GetOwner());
            auto &planetChunks = planetTerrain->m_chunks;
            // 1. Scan and expand.
            for (auto &chunk : planetChunks)
            {
                // futures.push_back(_PrimaryWorkers->Share([&, this](int id) { CheckLod(meshGenLock, chunk, planetInfo,
                // planetTransform, cameraLtw); }).share());
                CheckLod(meshGenLock, chunk, planetInfo, planetTransform, cameraLtw);
            }

            glm::mat4 matrix = glm::scale(
                glm::translate(glm::mat4_cast(planetTransform.GetRotation()), glm::vec3(planetTransform.GetPosition())),
                glm::vec3(1.0f));
            auto material = planetTerrain->m_surfaceMaterial.Get<Material>();
            if (material)
            {
                for (auto j = 0; j < planetChunks.size(); j++)
                {
                    RenderChunk(planetChunks[j], material, matrix, true);
                }
            }
        }
    }
}

void PlanetTerrainSystem::FixedUpdate()
{
}

void PlanetTerrainSystem::CheckLod(
    std::mutex &mutex,
    std::shared_ptr<TerrainChunk> &chunk,
    const PlanetInfo &info,
    const GlobalTransform &planetTransform,
    const GlobalTransform &cameraTransform)
{
    if (glm::distance(
            glm::dvec3(chunk->ChunkCenterPosition(
                planetTransform.GetPosition(), info.m_radius, planetTransform.GetRotation())),
            glm::dvec3(cameraTransform.GetPosition())) <
        info.m_lodDistance * info.m_radius / glm::pow(2, chunk->m_detailLevel + 1))
    {
        if (chunk->m_detailLevel < info.m_maxLodLevel)
        {
            chunk->Expand(mutex);
        }
    }
    if (chunk->m_c0)
        CheckLod(mutex, chunk->m_c0, info, planetTransform, cameraTransform);
    if (chunk->m_c1)
        CheckLod(mutex, chunk->m_c1, info, planetTransform, cameraTransform);
    if (chunk->m_c2)
        CheckLod(mutex, chunk->m_c2, info, planetTransform, cameraTransform);
    if (chunk->m_c3)
        CheckLod(mutex, chunk->m_c3, info, planetTransform, cameraTransform);
    if (glm::distance(
            glm::dvec3(chunk->ChunkCenterPosition(
                planetTransform.GetPosition(), info.m_radius, planetTransform.GetRotation())),
            glm::dvec3(cameraTransform.GetPosition())) >
        info.m_lodDistance * info.m_radius / glm::pow(2, chunk->m_detailLevel + 1))
    {
        chunk->Collapse();
    }
}

void PlanetTerrainSystem::RenderChunk(
    std::shared_ptr<TerrainChunk> &chunk,
    const std::shared_ptr<Material> &material,
    glm::mat4 &matrix,
    bool receiveShadow) const
{
    if (chunk->m_active) {
        const auto renderLayer = Application::GetLayer<RenderLayer>();
        renderLayer->DrawMesh(chunk->m_mesh, material, matrix, true);
    }
    if (chunk->m_childrenActive)
    {
        RenderChunk(chunk->m_c0, material, matrix, receiveShadow);
        RenderChunk(chunk->m_c1, material, matrix, receiveShadow);
        RenderChunk(chunk->m_c2, material, matrix, receiveShadow);
        RenderChunk(chunk->m_c3, material, matrix, receiveShadow);
    }
}
