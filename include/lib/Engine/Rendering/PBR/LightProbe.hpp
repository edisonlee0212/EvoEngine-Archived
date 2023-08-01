#pragma once
#include "Cubemap.hpp"

namespace EvoEngine
{
	class LightProbe : public IAsset
	{
		std::unique_ptr<Image> m_image = {};
		std::unique_ptr<ImageView> m_imageView = {};
		std::unique_ptr<Sampler> m_sampler = {};
		friend class RenderLayer;

		std::vector<std::shared_ptr<ImageView>> m_faceViews;
		std::vector<ImTextureID> m_imTextureIds;

	public:
		void Initialize(uint32_t resolution = 32);
		void ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap);

		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
	};
}