#pragma once
#include "AssetRef.hpp"
#include "ReflectionProbe.hpp"
#include "LightProbe.hpp"
namespace EvoEngine
{
    class EnvironmentalMap : public IAsset
    {
        friend class Graphics;
        friend class RenderLayer;
        friend class Resources;

        AssetRef m_lightProbe;
        AssetRef m_reflectionProbe;
    public:
        [[nodiscard]] bool IsReady() const;
        void ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap);
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
    };
}