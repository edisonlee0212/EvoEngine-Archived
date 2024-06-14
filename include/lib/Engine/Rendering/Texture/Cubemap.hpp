#pragma once
#include "Graphics.hpp"
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
#include "Texture2D.hpp"

namespace evo_engine
{
	class CubemapStorage;
	struct TextureStorageHandle;

	class Cubemap : public IAsset
	{
		friend class RenderLayer;

		friend class LightProbe;
		friend class ReflectionProbe;
		friend class TextureStorage;
		std::shared_ptr<TextureStorageHandle> m_textureStorageHandle;

	public:
		Cubemap();
		const CubemapStorage& PeekStorage() const;
		CubemapStorage& RefStorage() const;
		~Cubemap() override;
		void Initialize(uint32_t resolution, uint32_t mipLevels = 1);
		[[nodiscard]] uint32_t GetTextureStorageIndex() const;
		void ConvertFromEquirectangularTexture(const std::shared_ptr<Texture2D>& targetTexture);
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		[[nodiscard]] const std::shared_ptr<Image>& GetImage() const;
		[[nodiscard]] const std::shared_ptr<ImageView>& GetImageView() const;
		[[nodiscard]] const std::shared_ptr<Sampler>& GetSampler() const;
		[[nodiscard]] const std::vector<std::shared_ptr<ImageView>>& GetFaceViews() const;
	};
}
