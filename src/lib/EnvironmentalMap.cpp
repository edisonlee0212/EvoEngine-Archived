#include "EnvironmentalMap.hpp"
#include "EditorLayer.hpp"
#include "ProjectManager.hpp"

using namespace evo_engine;


void EnvironmentalMap::ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap)
{
    m_lightProbe = ProjectManager::CreateTemporaryAsset<LightProbe>();
    m_lightProbe.Get<LightProbe>()->ConstructFromCubemap(targetCubemap);
    m_reflectionProbe = ProjectManager::CreateTemporaryAsset<ReflectionProbe>();
    m_reflectionProbe.Get<ReflectionProbe>()->ConstructFromCubemap(targetCubemap);
}

void EnvironmentalMap::ConstructFromTexture2D(const std::shared_ptr<Texture2D>& targetTexture2D)
{
    auto cubemap = ProjectManager::CreateTemporaryAsset<Cubemap>();
    cubemap->ConvertFromEquirectangularTexture(targetTexture2D);
    m_lightProbe = ProjectManager::CreateTemporaryAsset<LightProbe>();
    m_lightProbe.Get<LightProbe>()->ConstructFromCubemap(cubemap);
    m_reflectionProbe = ProjectManager::CreateTemporaryAsset<ReflectionProbe>();
    m_reflectionProbe.Get<ReflectionProbe>()->ConstructFromCubemap(cubemap);
}

void EnvironmentalMap::ConstructFromRenderTexture(const std::shared_ptr<RenderTexture>& targetRenderTexture)
{
}

bool EnvironmentalMap::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    bool changed = false;
    static AssetRef targetTexture;

    if (editorLayer->DragAndDropButton<Cubemap>(targetTexture, "Convert from cubemap")) {
	    if (const auto tex = targetTexture.Get<Cubemap>()) {
            ConstructFromCubemap(tex);
            changed = true;
        }targetTexture.Clear();
    }

    if (editorLayer->DragAndDropButton<LightProbe>(m_lightProbe, "LightProbe")) changed = true;
    if (editorLayer->DragAndDropButton<LightProbe>(m_reflectionProbe, "ReflectionProbe")) changed = true;

    return changed;
}
