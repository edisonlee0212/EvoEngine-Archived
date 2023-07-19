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
		Image m_image = {};

		VkFormat m_imageFormat = VK_FORMAT_UNDEFINED;
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
