#pragma once
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
#include "Texture2D.hpp"

namespace EvoEngine
{
	class Cubemap : public IAsset
	{
		std::unique_ptr<Image> m_image = {};
		std::unique_ptr<ImageView> m_imageView = {};
		std::unique_ptr<Sampler> m_sampler = {};
		friend class RenderLayer;

		std::vector<std::shared_ptr<ImageView>> m_debugImageViews;
		std::vector<ImTextureID> m_imTextureIds;
	public:
		void ConvertFromEquirectangularTexture(const std::shared_ptr<Texture2D>& targetTexture);
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
	};
}
