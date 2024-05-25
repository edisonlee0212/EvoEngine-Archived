#pragma once
#include "GraphicsResources.hpp"

#include "Cubemap.hpp"
namespace EvoEngine
{
	struct TextureStorageHandle
	{
		int m_value = 0;
	};

	class Texture2DStorage
	{
		friend class TextureStorage;
		friend class Cubemap;
		std::vector<glm::vec4> m_newData;
		glm::uvec2 m_newResolution{};
		
	public:
		bool m_pendingDelete = false;

		std::shared_ptr<TextureStorageHandle> m_handle;

		std::shared_ptr<Image> m_image = {};
		std::shared_ptr<ImageView> m_imageView = {};
		std::shared_ptr<Sampler> m_sampler = {};
		ImTextureID m_imTextureId = VK_NULL_HANDLE;

		[[nodiscard]] VkImageLayout GetLayout() const;
		[[nodiscard]] VkImage GetVkImage() const;
		[[nodiscard]] VkImageView GetVkImageView() const;
		[[nodiscard]] VkSampler GetVkSampler() const;
		[[nodiscard]] std::shared_ptr<Image> GetImage() const;
		void Initialize(const glm::uvec2& resolution);
		void UploadData();
		void SetData(const std::vector<glm::vec4>& data, const glm::uvec2& resolution);
		void Clear();
	};
	class CubemapStorage
	{
	public:
		bool m_pendingDelete = false;

		std::shared_ptr<TextureStorageHandle> m_handle;

		std::shared_ptr<Image> m_image = {};
		std::shared_ptr<ImageView> m_imageView = {};
		std::shared_ptr<Sampler> m_sampler = {};
		void Clear();
		std::vector<std::shared_ptr<ImageView>> m_faceViews;
		std::vector<ImTextureID> m_imTextureIds;
		void Initialize(uint32_t resolution, uint32_t mipLevels);
		[[nodiscard]] VkImageLayout GetLayout() const;
		[[nodiscard]] VkImage GetVkImage() const;
		[[nodiscard]] VkImageView GetVkImageView() const;
		[[nodiscard]] VkSampler GetVkSampler() const;
		[[nodiscard]] std::shared_ptr<Image> GetImage() const;
	};
	class TextureStorage : public ISingleton<TextureStorage>
	{
		std::vector<Texture2DStorage> m_texture2Ds;
		std::vector<CubemapStorage> m_cubemaps;

		friend class RenderLayer;
		friend class Graphics;
		static void DeviceSync();
	public:
		static const Texture2DStorage& PeekTexture2DStorage(const std::shared_ptr<TextureStorageHandle>& handle);
		static Texture2DStorage& RefTexture2DStorage(const std::shared_ptr<TextureStorageHandle>& handle);

		static const CubemapStorage& PeekCubemapStorage(const std::shared_ptr<TextureStorageHandle>& handle);
		static CubemapStorage& RefCubemapStorage(const std::shared_ptr<TextureStorageHandle>& handle);

		static void UnRegisterTexture2D(const std::shared_ptr<TextureStorageHandle>& handle);
		static void UnRegisterCubemap(const std::shared_ptr<TextureStorageHandle>& handle);
		static std::shared_ptr<TextureStorageHandle> RegisterTexture2D();
		static std::shared_ptr<TextureStorageHandle> RegisterCubemap();
		static void Initialize();
	};
}
