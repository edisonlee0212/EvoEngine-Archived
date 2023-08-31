#pragma once
#include "Graphics.hpp"
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
#include "Texture2D.hpp"

namespace EvoEngine
{
	class Cubemap : public IAsset
	{
		std::shared_ptr<Image> m_image = {};
		std::shared_ptr<ImageView> m_imageView = {};
		std::shared_ptr<Sampler> m_sampler = {};
		friend class RenderLayer;


		std::vector<std::shared_ptr<ImageView>> m_faceViews;
		std::vector<ImTextureID> m_imTextureIds;
		friend class LightProbe;
		friend class ReflectionProbe;
		friend class TextureStorage;
		uint32_t m_textureStorageIndex = UINT32_MAX;
	public:
		~Cubemap() override;
		void Initialize(uint32_t resolution, uint32_t mipLevels = 1);
		[[nodiscard]] uint32_t GetTextureStorageIndex() const;
		void ConvertFromEquirectangularTexture(const std::shared_ptr<Texture2D>& targetTexture);
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		[[nodiscard]] const std::shared_ptr<Image>& GetImage() const;
		[[nodiscard]] const std::shared_ptr<ImageView>& GetImageView() const;
		[[nodiscard]] const std::shared_ptr<Sampler>& GetSampler() const;
		[[nodiscard]] const std::vector<std::shared_ptr<ImageView>>& GetFaceViews() const;
	};
}
