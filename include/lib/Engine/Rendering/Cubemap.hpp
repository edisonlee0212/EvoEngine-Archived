#pragma once
#include "Graphics.hpp"
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


		std::vector<std::shared_ptr<ImageView>> m_faceViews;
		std::vector<ImTextureID> m_imTextureIds;
		friend class LightProbe;
		friend class ReflectionProbe;
	public:
		void Initialize(uint32_t resolution);

		void ConvertFromEquirectangularTexture(const std::shared_ptr<Texture2D>& targetTexture);
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		[[nodiscard]] const std::unique_ptr<Image>& GetImage() const;
		[[nodiscard]] const std::unique_ptr<ImageView>& GetImageView() const;
		[[nodiscard]] const std::unique_ptr<Sampler>& GetSampler() const;
		[[nodiscard]] const std::vector<std::shared_ptr<ImageView>>& GetFaceViews() const;
	};
}
