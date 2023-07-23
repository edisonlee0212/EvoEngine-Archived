#pragma once
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
namespace EvoEngine
{
	enum class TextureColorType {
		Red = 1,
		RG = 2,
		RGB = 3,
		RGBA = 4
	};

	class Texture2D : public IAsset
	{
		friend class EditorLayer;
		std::unique_ptr<Image> m_image = {};
		std::unique_ptr<ImageView> m_imageView = {};
		std::unique_ptr<Sampler> m_sampler = {};

		VkFormat m_imageFormat = VK_FORMAT_UNDEFINED;

		ImTextureID m_imTextureId = VK_NULL_HANDLE;
	protected:
		bool SaveInternal(const std::filesystem::path& path) override;
		bool LoadInternal(const std::filesystem::path& path) override;
	public:
		float m_gamma = 1.0f;
		[[nodiscard]] glm::vec2 GetResolution() const;
		void StoreToPng(
			const std::string& path,
			int resizeX = -1,
			int resizeY = -1,
			bool alphaChannel = false,
			unsigned compressionLevel = 8) const;
		void StoreToJpg(const std::string& path, int resizeX = -1, int resizeY = -1, unsigned quality = 100) const;
		void StoreToHdr(const std::string& path, int resizeX = -1, int resizeY = -1,
			bool alphaChannel = false, unsigned quality = 100) const;
		ImTextureID GetImTextureId() const;
	};
}
