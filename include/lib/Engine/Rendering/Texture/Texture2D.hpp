#pragma once
#include "GraphicsResources.hpp"
#include "IAsset.hpp"

namespace EvoEngine
{
	class Texture2DStorage;
	struct TextureStorageHandle;

	enum class TextureColorType {
		Red = 1,
		RG = 2,
		RGB = 3,
		RGBA = 4
	};

	class Texture2D : public IAsset
	{
		friend class EditorLayer;
		friend class Resources;
		friend class Cubemap;
		friend class TextureStorage;
		friend class RenderLayer;
		
		std::shared_ptr<TextureStorageHandle> m_textureStorageHandle;
		

		void SetData(const std::vector<glm::vec4>& data, const glm::uvec2& resolution) const;
		
	protected:
		bool SaveInternal(const std::filesystem::path& path) const override;
		bool LoadInternal(const std::filesystem::path& path) override;
	public:
		void ApplyOpacityMap(const std::shared_ptr<Texture2D>& target);

		void Serialize(YAML::Emitter& out) const override;
		void Deserialize(const YAML::Node& in) override;
		bool m_hdr = false;
		Texture2D();
		const Texture2DStorage& PeekTexture2DStorage() const;
		Texture2DStorage& RefTexture2DStorage() const;
		[[nodiscard]] VkImageLayout GetLayout() const;
		[[nodiscard]] VkImage GetVkImage() const;
		[[nodiscard]] VkImageView GetVkImageView() const;
		[[nodiscard]] VkSampler GetVkSampler() const;
		[[nodiscard]] std::shared_ptr<Image> GetImage() const;
		ImTextureID GetImTextureId() const;
		[[nodiscard]] uint32_t GetTextureStorageIndex() const;
		~Texture2D() override;
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		[[nodiscard]] glm::ivec2 GetResolution() const;
		void StoreToPng(
			const std::string& path,
			int resizeX = -1,
			int resizeY = -1,
			unsigned compressionLevel = 8) const;
		void StoreToTga(
			const std::string& path,
			int resizeX = -1,
			int resizeY = -1) const;
		void StoreToJpg(const std::string& path, int resizeX = -1, int resizeY = -1, unsigned quality = 100) const;
		void StoreToHdr(const std::string& path, int resizeX = -1, int resizeY = -1,
			bool alphaChannel = false, unsigned quality = 100) const;
		
		void GetRgbaChannelData(std::vector<glm::vec4>& dst, int resizeX = -1, int resizeY = -1) const;
		void GetRgbChannelData(std::vector<glm::vec3>& dst, int resizeX = -1, int resizeY = -1) const;
		void GetRgChannelData(std::vector<glm::vec2>& dst, int resizeX = -1, int resizeY = -1) const;
		void GetRedChannelData(std::vector<float>& dst, int resizeX = -1, int resizeY = -1) const;

		void SetRgbaChannelData(const std::vector<glm::vec4>& src, const glm::uvec2& resolution);
		void SetRgbChannelData(const std::vector<glm::vec3>& src, const glm::uvec2& resolution);
		void SetRgChannelData(const std::vector<glm::vec2>& src, const glm::uvec2& resolution);
		void SetRedChannelData(const std::vector<float>& src, const glm::uvec2& resolution);
	};
}
