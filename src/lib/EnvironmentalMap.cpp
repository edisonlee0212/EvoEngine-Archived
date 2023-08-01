#include "EnvironmentalMap.hpp"
#include "EditorLayer.hpp"
#include "ProjectManager.hpp"

using namespace EvoEngine;


void EnvironmentalMap::ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap)
{
    m_lightProbe = ProjectManager::CreateTemporaryAsset<LightProbe>();
    m_lightProbe.Get<LightProbe>()->ConstructFromCubemap(targetCubemap);
    m_reflectionProbe = ProjectManager::CreateTemporaryAsset<ReflectionProbe>();
    m_reflectionProbe.Get<ReflectionProbe>()->ConstructFromCubemap(targetCubemap);
}

void EnvironmentalMap::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    static AssetRef targetTexture;

    if (editorLayer->DragAndDropButton<Cubemap>(targetTexture, "Convert from cubemap")) {
        auto tex = targetTexture.Get<Cubemap>();
        if (tex)
            ConstructFromCubemap(tex);
        targetTexture.Clear();
    }

    editorLayer->DragAndDropButton<LightProbe>(m_lightProbe, "LightProbe");
    editorLayer->DragAndDropButton<LightProbe>(m_reflectionProbe, "ReflectionProbe");
}